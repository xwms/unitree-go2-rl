#!/usr/bin/env bash
# 本地一键部署（仿真/开发，不含真机 DDS）：
#   uv 建 3.11 venv → C++ SDK 装入 .venv（无需 sudo）→ uv sync --extra sim（MuJoCo）→ 冒烟测试
# 真机通信（cyclonedds / unitree-sdk2py）需额外运行 scripts/bootstrap_robot.sh
# 用法: bash scripts/bootstrap_local.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1. Python 3.11 + venv
uv python install 3.11
if [[ ! -d .venv ]]; then
    uv venv --python 3.11 .venv
fi

# 2. C++ SDK 安装到 .venv（CMake 只编译示例并安装预编译静态库）
#    如需 C++ 示例可把 BUILD_EXAMPLES 改为 ON
cmake -S stack/unitree_sdk2 -B stack/unitree_sdk2/build \
    -DCMAKE_INSTALL_PREFIX="$ROOT/.venv" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=OFF
cmake --build stack/unitree_sdk2/build -j"$(nproc)"
cmake --install stack/unitree_sdk2/build

# 3. 安装仿真依赖
uv sync --extra sim

# 4. 冒烟测试
bash scripts/smoke_test.sh

echo "[OK] 本地部署完成（真机通信请再运行 scripts/bootstrap_robot.sh）"
