#!/usr/bin/env bash
# 基线自动记录：包装 train_stair.sh，后台采样 GPU/CPU，结束后汇总关键指标
# 用法: bash scripts/record_baseline.sh [train_stair.sh 参数，如 --stage1-only --max_iterations 10]
# 结果在 logs/baseline/<时间戳>/（train.log + gpu.csv + cpu.csv + summary.txt）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/logs/baseline/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$OUT"
GPU_CSV="$OUT/gpu.csv"
CPU_CSV="$OUT/cpu.csv"
echo "timestamp,utilization.gpu,memory.used_mb" > "$GPU_CSV"
echo "epoch,utilization.cpu_total,utilization.cpu_maxcore" > "$CPU_CSV"

# 后台采样循环
(
    while true; do
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
            --format=csv,noheader,nounits >> "$GPU_CSV"
        # 每核利用率：/proc/stat 两次采样差值（busy = user+nice+system+irq+softirq+steal）
        STAT1=$(mktemp); STAT2=$(mktemp)
        awk 'match($1, /^cpu[0-9]+$/) { b=$2+$3+$4+$7+$8+$9; print $1, b, b+$5+$6 }' \
            /proc/stat > "$STAT1"
        sleep 0.5
        awk 'match($1, /^cpu[0-9]+$/) { b=$2+$3+$4+$7+$8+$9; print $1, b, b+$5+$6 }' \
            /proc/stat > "$STAT2"
        paste "$STAT1" "$STAT2" | awk -v t="$(date +%s)" '
            { du = ($5 - $2) / ($6 - $3) * 100; tot += du; n++; if (du > mx) mx = du }
            END { printf "%d,%.1f,%.1f\n", t, tot / n, mx }' >> "$CPU_CSV"
        rm -f "$STAT1" "$STAT2"
        sleep 1.5
    done
) &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT

echo "[record_baseline] 结果目录: $OUT"
echo "[record_baseline] 训练输出见: $OUT/train.log"

PYTHONUNBUFFERED=1 bash "$ROOT/scripts/train_stair.sh" "$@" 2>&1 \
    | while IFS= read -r line; do printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$line"; done \
    | tee "$OUT/train.log"

kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true
trap - EXIT

"$ROOT/.venv/bin/python" - "$OUT" <<'PY'
import csv
import re
import sys
from datetime import datetime

out = sys.argv[1]

utils, mems = [], []
with open(f"{out}/gpu.csv") as f:
    for row in csv.DictReader(f):
        utils.append(float(row["utilization.gpu"]))
        mems.append(float(row["memory.used_mb"]))
cpus, cores = [], []
with open(f"{out}/cpu.csv") as f:
    for row in csv.DictReader(f):
        cpus.append(float(row["utilization.cpu_total"]))
        cores.append(float(row["utilization.cpu_maxcore"]))

ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)")
iter_re = re.compile(r"Learning iteration (\d+)/")
iter_time_re = re.compile(r"Iteration time:\s*([0-9.]+)s")
first_ts, iter_start_ts, iter_times = None, None, []
for line in open(f"{out}/train.log", errors="ignore"):
    m = ts_re.match(line)
    if m:
        t = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ")
        if first_ts is None:
            first_ts = t
        if iter_start_ts is None and iter_re.search(line):
            iter_start_ts = t
    m = iter_time_re.search(line)
    if m:
        iter_times.append(float(m.group(1)))

def stats(xs):
    return (sum(xs) / len(xs), max(xs)) if xs else (0.0, 0.0)

MEM_OFFSET_MB = 5120  # 共享卡固定底噪，从读数中扣除

lines = []
lines.append("== 基线汇总 ==")
lines.append(f"样本数: gpu={len(utils)}  cpu={len(cpus)}")
g_avg, g_peak = stats(utils)
lines.append(f"GPU 利用率: 平均 {g_avg:.1f}%  峰值 {g_peak:.1f}%")
lines.append(f"显存占用: 峰值 {(max(mems) - MEM_OFFSET_MB) / 1024:.2f} GB（已扣 5GB 底噪）" if mems else "显存: 无数据")
t_avg, t_peak = stats(cpus)
lines.append(f"CPU 整体(全核平均): 平均 {t_avg:.1f}%  峰值 {t_peak:.1f}%")
c_avg, c_peak = stats(cores)
lines.append(f"CPU 单核最高: 平均 {c_avg:.1f}%  峰值 {c_peak:.1f}%")
if first_ts and iter_start_ts:
    lines.append(f"首轮启动耗时(日志首行→首个iter): {(iter_start_ts - first_ts).total_seconds():.0f} s")
if iter_times:
    lines.append(f"首个迭代耗时(含编译预热): {iter_times[0]:.1f} s")
    rest = iter_times[1:]
    if rest:
        lines.append(
            f"每迭代耗时: 中位 {sorted(rest)[len(rest) // 2]:.1f} s  均值 {sum(rest) / len(rest):.1f} s  (共 {len(rest)} 个迭代)"
        )
else:
    lines.append("每迭代耗时: 未找到 Iteration time 记录")
lines.append(f"原始数据: {out}/")

summary = "\n".join(lines)
print(summary)
with open(f"{out}/summary.txt", "w") as f:
    f.write(summary + "\n")
PY
