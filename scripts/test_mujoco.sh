#!/usr/bin/env bash
# MuJoCo 启动测试：加载 Go2 场景并仿真
# 用法:
#   bash scripts/test_mujoco.sh             # 可视化窗口（需要图形界面；右键旋转视角、滚轮缩放）
#   bash scripts/test_mujoco.sh --headless  # 无头物理测试（SSH 无显示时用）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--headless" ]]; then
    .venv/bin/python - <<'PY'
import mujoco, time
m = mujoco.MjModel.from_xml_path("stack/unitree_mujoco/unitree_robots/go2/scene.xml")
d = mujoco.MjData(m)
print(f"[OK] 模型加载: {m.nbody} bodies, {m.njnt} joints, timestep={m.opt.timestep}")
t0 = time.time()
for _ in range(1000):
    mujoco.mj_step(m, d)
print(f"[OK] 仿真 1000 步耗时 {time.time()-t0:.2f}s, 底座高度 z = {d.qpos[2]:.3f}")
PY
else
    .venv/bin/python - <<'PY'
import mujoco, mujoco.viewer, time
m = mujoco.MjModel.from_xml_path("stack/unitree_mujoco/unitree_robots/go2/scene.xml")
d = mujoco.MjData(m)
viewer = mujoco.viewer.launch_passive(m, d)
print("[OK] MuJoCo 窗口已打开，关闭窗口退出")
while viewer.is_running():
    mujoco.mj_step(m, d)
    viewer.sync()
    time.sleep(0.002)
viewer.close()
PY
fi
