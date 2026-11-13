#!/usr/bin/env bash
# 真机通信部署（DDS）：在本地部署基础上，编译 C 版 CycloneDDS 并安装 unitree-sdk2py
# cyclonedds 0.10.2 无 cp311 wheel，Python 侧走源码构建，需要本脚本编译的 C 库
# 用法: bash scripts/bootstrap_robot.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1. 基础部署（venv + C++ SDK + 仿真依赖）
bash scripts/bootstrap_local.sh

# 2. 编译 C 版 CycloneDDS 0.10.2 到 .venv 前缀
CYCLONE_SRC="$ROOT/stack/cyclonedds"
if [[ ! -f "$ROOT/.venv/lib/libddsc.so" ]]; then
    if [[ ! -d "$CYCLONE_SRC" ]]; then
        mkdir -p "$CYCLONE_SRC"
        curl -sL https://github.com/eclipse-cyclonedds/cyclonedds/archive/refs/tags/0.10.2.tar.gz \
            | tar xz -C "$CYCLONE_SRC" --strip-components=1
    fi
    cmake -S "$CYCLONE_SRC" -B "$CYCLONE_SRC/build" \
        -DCMAKE_INSTALL_PREFIX="$ROOT/.venv" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTING=OFF
    cmake --build "$CYCLONE_SRC/build" -j"$(nproc)"
    cmake --install "$CYCLONE_SRC/build"
else
    echo "[skip] C 版 CycloneDDS 已安装"
fi

# 3. 记录 CYCLONEDDS_HOME（供 uv 构建 cyclonedds sdist 时使用，路径为本机绝对路径）
export CYCLONEDDS_HOME="$ROOT/.venv"
echo "CYCLONEDDS_HOME=$ROOT/.venv" > .env

# 4. 安装真机通信依赖（unitree-sdk2py + cyclonedds 源码构建 + opencv）
uv sync --extra robot

# 5. DDS 收发冒烟（用 Go2 LowCmd_ 真实消息类型）
.venv/bin/python - <<'PY'
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from cyclonedds.pub import DataWriter
from cyclonedds.util import duration
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

dp = DomainParticipant(0)
tp = Topic(dp, "go2_smoke_test", LowCmd_)
dr = DataReader(dp, tp)
dw = DataWriter(dp, tp)
dw.write(unitree_go_msg_dds__LowCmd_())
sample = dr.take_one(timeout=duration(seconds=2))
assert sample is not None, "DDS 读写超时"
print("[OK] DDS 收发正常")
PY

echo "[OK] 真机部署完成"
