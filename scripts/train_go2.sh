#!/usr/bin/env bash
# Go2 训练 / 演示入口（云端 Isaac Lab）
# 用法:
#   bash scripts/train_go2.sh                                # 训练 Unitree-Go2-Velocity（headless）
#   bash scripts/train_go2.sh --max_iterations 100           # 自定义参数
#   TASK=Unitree-Go2-Velocity bash scripts/train_go2.sh      # 换任务
#   bash scripts/train_go2.sh play                           # 演示已训练模型（需要显示，可加 --headless --video）
#   bash scripts/train_go2.sh play --checkpoint <ckpt文件>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TASK="${TASK:-Unitree-Go2-Velocity}"

if [[ "${1:-}" == "play" ]]; then
    shift
    exec "$ROOT/.venv/bin/python" "$ROOT/stack/unitree_rl_lab/scripts/rsl_rl/play.py" --task "$TASK" "$@"
fi

exec "$ROOT/.venv/bin/python" "$ROOT/stack/unitree_rl_lab/scripts/rsl_rl/train.py" --task "$TASK" --headless "$@"
