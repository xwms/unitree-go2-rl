#!/usr/bin/env bash
# 本地冒烟测试（仿真）：Python 依赖 + MuJoCo 物理仿真 + C++ SDK 已装进 .venv
# 用法: bash scripts/smoke_test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

# 1. Python 依赖 + MuJoCo 物理仿真
"$PY" - <<'PY'
import numpy, mujoco
print("[OK] python deps:", "numpy", numpy.__version__, "| mujoco", mujoco.__version__)
m = mujoco.MjModel.from_xml_path("stack/unitree_mujoco/unitree_robots/go2/scene.xml")
d = mujoco.MjData(m)
for _ in range(100):
    mujoco.mj_step(m, d)
print("[OK] MuJoCo 仿真 100 步正常")
PY

# 2. C++ SDK：静态库 + cmake config 应已安装到 .venv 前缀
if find "$ROOT/.venv" \( -name "libunitree_sdk2*" -o -name "unitree_sdk2Config.cmake" \) | grep . >/dev/null; then
    echo "[OK] C++ SDK 已安装到 .venv"
else
    echo "[FAIL] 未找到 libunitree_sdk2，请先运行 bootstrap_local.sh" >&2
    exit 1
fi

echo "[OK] 冒烟测试全部通过"
