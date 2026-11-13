#!/usr/bin/env bash
# 云端一键部署（Isaac Lab / Isaac Sim 训练机）：
# Python 3.11 venv + 全部依赖（含 isaaclab 2.3.0 + isaacsim 5.1.0）+ unitree_model USD 模型
# 注意：Isaac 栈约需 30G+ 磁盘；需 NVIDIA 驱动 + CUDA（如 PRO 6000）
# 用法: bash scripts/bootstrap_cloud.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 云端 shell 里可能挂着 conda 环境，先摘掉以免干扰 uv
unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV 2>/dev/null || true
# 本地 .env 里的 CYCLONEDDS_HOME 是本地路径，云端用不到，删掉避免污染
rm -f .env

# 1. Python 3.11 + venv + 全部依赖（Isaac Lab 2.3.0 / Isaac Sim 5.1.0 只有 cp311 wheel）
uv python install 3.11
if [[ ! -d .venv ]]; then
    uv venv --python 3.11 .venv
fi
uv sync --extra rl

# 2. 下载 unitree_model（USD 模型，已迁移到 HuggingFace datasets）
#    git clone 只拉 LFS 指针文件，必须用 hf download --include 拉真实内容
MODEL_DIR="$ROOT/stack/unitree_model"
if [[ -d "$MODEL_DIR/Go2" ]]; then
    echo "[skip] 模型已存在: $MODEL_DIR"
else
    rm -rf "$MODEL_DIR"
    uvx --from huggingface_hub hf download unitreerobotics/unitree_model \
        --repo-type dataset --local-dir "$MODEL_DIR" --include "Go2/*"
fi

# 3. 把 UNITREE_MODEL_DIR 写入 unitree_rl_lab 配置（幂等，可重复执行）
UNITREE_PY="$ROOT/stack/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py"
sed -i "s|^UNITREE_MODEL_DIR = .*|UNITREE_MODEL_DIR = \"$MODEL_DIR\"  # auto-configured by bootstrap_cloud.sh|" "$UNITREE_PY"
grep "^UNITREE_MODEL_DIR" "$UNITREE_PY"

# 4. 验证导入
"$ROOT/.venv/bin/python" -c "import torch, isaaclab; from importlib.metadata import version; print('[OK] torch', torch.__version__, '| isaaclab', version('isaaclab'))"

echo "[OK] 云端部署完成。训练: bash scripts/train_go2.sh"
