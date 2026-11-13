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
│   └── train_go2.sh        # 云端训练 / 演示入口
└── stack/
    ├── unitree_sdk2/           # 宇树 C++ SDK
    ├── unitree_sdk2_python/    # 宇树 Python SDK（editable 安装，仅真机用）
    ├── unitree_mujoco/         # 宇树 MuJoCo 模型与示例
    ├── unitree_rl_lab/         # 宇树官方 Isaac Lab 训练库（editable 安装）
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

```bash
bash scripts/bootstrap_cloud.sh    # 安装 rl extra + 下载 unitree_model + 写入 UNITREE_MODEL_DIR
bash scripts/train_go2.sh          # 训练 Unitree-Go2-Velocity（headless）
bash scripts/train_go2.sh play     # 演示已训练模型（需要显示环境）
```

## 注意事项

- 只做训练/仿真时直接 `uv sync` 即可，不涉及 DDS；只有 `robot` extra
  （cyclonedds 无 cp311 wheel，需先编译 C 库并设置 CYCLONEDDS_HOME）必须走
  `bootstrap_robot.sh`。
- 云端训练需要 HuggingFace 的 `unitreerobotics/unitree_model`（公开模型，无需 token），
  由 bootstrap_cloud 自动下载到 `stack/unitree_model/`。
