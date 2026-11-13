#!/usr/bin/env bash
# 楼梯攀爬两阶段训练入口（云端 Isaac Lab，论文 arXiv 2602.03087v2）
# 用法:
#   bash scripts/train_stair.sh                        # Stage1 训练完自动用最新 checkpoint 续训 Stage2
#   bash scripts/train_stair.sh --stage1-only          # 只训 Stage1（金字塔平台导航）
#   bash scripts/train_stair.sh --stage2-only          # 只用 Stage1 最新 checkpoint 训 Stage2
#   bash scripts/train_stair.sh --max_iterations 5000  # 其余参数透传给 train.py
#   TASK=Unitree-Go2-Stair-Stage2 bash scripts/train_stair.sh play   # 演示（需显示，可加 --video）
#   TASK=Unitree-Go2-Stair-Stage2 bash scripts/train_stair.sh eval --checkpoint <ckpt>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
RSL_RL_DIR="$ROOT/stack/unitree_rl_lab/scripts/rsl_rl"
EVAL_PY="$ROOT/src/go2_stair/go2_stair/scripts/eval_stair.py"
STAGE1_TASK="Unitree-Go2-Stair-Stage1"
STAGE2_TASK="Unitree-Go2-Stair-Stage2"
STAGE1_EXP="go2_stair_stage1"

latest_run() {
    ls -dt "$ROOT/logs/rsl_rl/$1"/*/ 2>/dev/null | head -n 1 | sed 's:/$::'
}

MODE="train"
STAGE1_ONLY=false
STAGE2_ONLY=false
EXTRA=()
for arg in "$@"; do
    case "$arg" in
        --stage1-only) STAGE1_ONLY=true ;;
        --stage2-only) STAGE2_ONLY=true ;;
        *) EXTRA+=("$arg") ;;
    esac
done
set -- "${EXTRA[@]}"

if [[ "${1:-}" == "play" ]]; then
    shift
    TASK="${TASK:-$STAGE2_TASK}"
    exec "$PY" "$RSL_RL_DIR/play.py" --task "$TASK" "$@"
fi

if [[ "${1:-}" == "eval" ]]; then
    shift
    TASK="${TASK:-$STAGE2_TASK}"
    exec "$PY" "$EVAL_PY" --task "$TASK" --headless "$@"
fi

if ! $STAGE2_ONLY; then
    echo "==> Stage1 训练: $STAGE1_TASK"
    "$PY" "$RSL_RL_DIR/train.py" --task "$STAGE1_TASK" --headless "$@"
    if $STAGE1_ONLY; then
        exit 0
    fi
fi

STAGE1_RUN="$(latest_run "$STAGE1_EXP")"
if [[ -z "$STAGE1_RUN" ]]; then
    echo "错误: 未找到 Stage1 训练记录 (logs/rsl_rl/$STAGE1_EXP)" >&2
    exit 1
fi
echo "==> Stage2 训练: $STAGE2_TASK（从 Stage1 checkpoint 续训: $STAGE1_RUN）"
exec "$PY" "$RSL_RL_DIR/train.py" --task "$STAGE2_TASK" --headless --resume --load_run "$STAGE1_RUN" "$@"
