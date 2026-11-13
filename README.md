# Go2 训练与仿真一键部署

宇树 Go2 的训练与仿真部署包：uv 管理、单一 uv.lock、统一 Python 3.11。
本地负责真机通信 + MuJoCo 仿真，云端负责 Isaac Lab / Isaac Sim 训练。

```
unitree-go2-rl/
├── pyproject.toml          # 依赖定义（sim / robot / rl 三个 extra）
├── uv.lock                 # 已锁定的依赖，无需重新解析
├── scripts/
│   ├── bootstrap_local.sh  # 本地一键部署（仿真，不含 DDS）
│   ├── bootstrap_robot.sh  # 真机通信部署（编译 C 版 CycloneDDS + unitree-sdk2py）
│   ├── bootstrap_cloud.sh  # 云端一键部署（Isaac 栈）
│   ├── smoke_test.sh       # 本地冒烟测试（MuJoCo 仿真 + C++ SDK）
│   ├── test_mujoco.sh      # MuJoCo 启动测试（可视化 / --headless）
│   ├── train_stair.sh      # 楼梯两阶段训练 / 演示 / 评估入口（云端）
│   ├── record_baseline.sh  # 训练时采样 GPU/CPU/显存，结束后出基线汇总
│   ├── register_bridge.py  # 楼梯任务注册桥（部署时拷贝到 unitree_rl_lab 的 go2 包）
│   └── train_go2.sh        # 云端训练 / 演示入口
└── stack/
    ├── unitree_sdk2/           # 宇树 C++ SDK
    ├── unitree_sdk2_python/    # 宇树 Python SDK（editable 安装，仅真机用）
    ├── unitree_mujoco/         # 宇树 MuJoCo 模型与示例
    ├── unitree_rl_lab/         # 宇树官方 Isaac Lab 训练库（editable 安装）
    ├── unitree_model/          # Go2 USD 模型（bootstrap_cloud 自动下载）
    └── cyclonedds/             # C 版 CycloneDDS 0.10.2 源码（bootstrap_robot 自动下载）
```

## 前提条件

- Linux x86_64（unitree_sdk2 的静态库只有 Linux 版）
- [uv](https://docs.astral.sh/uv/)：`curl -LsSf https://astral.sh/uv/install.sh | sh`
- 真机部署：`sudo apt install cmake build-essential`（编译 C 版 CycloneDDS 用）
- 云端机器：NVIDIA 驱动 + CUDA（Isaac Sim 要求）

## 本地部署（MuJoCo 仿真）

```bash
bash scripts/bootstrap_local.sh
```

部署完成后：

```bash
bash scripts/smoke_test.sh              # 复测：MuJoCo 仿真 + C++ SDK
bash scripts/test_mujoco.sh             # MuJoCo 可视化窗口（右键旋转视角、滚轮缩放）
bash scripts/test_mujoco.sh --headless  # 无头物理测试（SSH 无显示时用）
```

## 真机通信部署（DDS）

需要跟真机收发消息时才装，会编译 C 版 CycloneDDS 并安装 unitree-sdk2py：

```bash
bash scripts/bootstrap_robot.sh
```

## 云端部署（Isaac Lab 训练）

从空机器开始（`stack/` 是 gitignored，需手动补）：

```bash
# 1. 克隆主仓库
git clone https://github.com/<你的>/unitree-go2-rl.git
cd unitree-go2-rl

# 2. 补 stack 依赖（unitree_rl_lab / unitree_sdk2_python；模型由 bootstrap 自动下载）
git clone https://github.com/unitreerobotics/unitree_rl_lab.git stack/unitree_rl_lab
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git stack/unitree_sdk2_python

# 3. 写楼梯任务注册桥（覆盖 unitree_rl_lab 的 go2 注册文件）
cp scripts/register_bridge.py \
   stack/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/__init__.py

# 4. 一键部署（建 venv + uv sync + 下载 unitree_model + 写 UNITREE_MODEL_DIR + 验证）
bash scripts/bootstrap_cloud.sh

# 5. 验证任务注册
.venv/bin/python stack/unitree_rl_lab/scripts/rsl_rl/list_envs.py
```

看到 `Unitree-Go2-Stair-Stage1/Stage2` 即部署成功。训练（楼梯两阶段任务）：

```bash
bash scripts/train_stair.sh --num_envs 8192 --max_iterations 3000   # Stage1 训完自动续训 Stage2
bash scripts/train_stair.sh --stage1-only --max_iterations 500      # 只训 Stage1
bash scripts/train_stair.sh play                                    # 演示（需显示环境）
```

> 说明：`uv sync` 只装 Python 包；Go2 的 USD 模型（非 pip 数据）和 `UNITREE_MODEL_DIR`
> 路径配置由 `bootstrap_cloud.sh` 负责，缺一训练会起不来。

## 注意事项

- 只做训练/仿真时直接 `uv sync` 即可，不涉及 DDS；只有 `robot` extra
  （cyclonedds 无 cp311 wheel，需先编译 C 库并设置 CYCLONEDDS_HOME）必须走
  `bootstrap_robot.sh`。
- 云端训练需要 HuggingFace 的 `unitreerobotics/unitree_model`（公开模型，无需 token），
  由 bootstrap_cloud 自动下载到 `stack/unitree_model/`。
- 云端 GPU 是 Blackwell（sm_120），torch 固定 cu128 构建（来自 download.pytorch.org，
  见 pyproject 的 `pytorch-cu128` 索引）；不要在 PyPI 装普通 torch，否则报
  "no kernel image is available"。
- `num_envs` 甜点是 8192（RTX PRO 6000 96GB）：4096 没吃满 GPU、16384 无额外收益。
- 共享云容器可能被回收，长训务必挂 tmux，并用 `--resume` 留续训余地。
