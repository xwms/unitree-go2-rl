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

PYTHONUNBUFFERED=1 bash "$ROOT/scripts/train_stair.sh" "$@" 2>&1 | tee "$OUT/train.log"

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

ts_re = re.compile(r"\[(\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
iter_re = re.compile(r"\biter (\d+)/")
first_ts, iter_ts = None, []
for line in open(f"{out}/train.log", errors="ignore"):
    m = ts_re.search(line)
    if not m:
        continue
    t = datetime.strptime(m.group(1), "%m-%d %H:%M:%S")
    if first_ts is None:
        first_ts = t
    if iter_re.search(line):
        iter_ts.append(t)

def stats(xs):
    return (sum(xs) / len(xs), max(xs)) if xs else (0.0, 0.0)

lines = []
lines.append("== 基线汇总 ==")
lines.append(f"样本数: gpu={len(utils)}  cpu={len(cpus)}")
g_avg, g_peak = stats(utils)
lines.append(f"GPU 利用率: 平均 {g_avg:.1f}%  峰值 {g_peak:.1f}%")
lines.append(f"显存占用: 峰值 {max(mems) / 1024:.2f} GB" if mems else "显存: 无数据")
t_avg, t_peak = stats(cpus)
lines.append(f"CPU 整体(全核平均): 平均 {t_avg:.1f}%  峰值 {t_peak:.1f}%")
c_avg, c_peak = stats(cores)
lines.append(f"CPU 单核最高: 平均 {c_avg:.1f}%  峰值 {c_peak:.1f}%")
if first_ts and iter_ts:
    lines.append(f"首轮启动耗时(日志首行→首个iter): {(iter_ts[0] - first_ts).total_seconds():.0f} s")
if len(iter_ts) >= 2:
    gaps = [(iter_ts[i + 1] - iter_ts[i]).total_seconds() for i in range(len(iter_ts) - 1)]
    lines.append(f"每迭代耗时: 中位 {sorted(gaps)[len(gaps) // 2]:.1f} s  均值 {sum(gaps) / len(gaps):.1f} s  (共 {len(gaps)} 个间隔)")
else:
    lines.append("每迭代耗时: iter 记录不足 2 条，无法计算")
lines.append(f"原始数据: {out}/")

summary = "\n".join(lines)
print(summary)
with open(f"{out}/summary.txt", "w") as f:
    f.write(summary + "\n")
PY
