"""Replay the exported stair policy in MuJoCo (visual demo or headless check).

Builds a straight staircase matching the training terrain (12 steps, configurable step
height, 1.5 m top landing), assembles the same 522-dim observation the policy was trained
on (45 proprio + 4 goal + 32 foot heights + 441 height map, with the training scales and
clips) and drives the exported ONNX policy at 50 Hz with a torque-level PD.

Usage:
    .venv/bin/python scripts/play_stair_mujoco.py                 # viewer window
    .venv/bin/python scripts/play_stair_mujoco.py --headless      # print progress only
    .venv/bin/python scripts/play_stair_mujoco.py --step_height 0.06 --duration 15
"""

import argparse
import math
import os

import mujoco
import numpy as np
import onnxruntime as ort

# --- training-consistent constants (from deploy/stair/deploy.yaml and the task cfg) ---
# policy action/observation order = the Isaac asset joint order [FL, FR, RL, RR] x
# [hip, thigh, calf], with the defaults already listed in that same (asset) order
ASSET_JOINTS = [
    ("FL_hip_joint", 0.1),
    ("FR_hip_joint", -0.1),
    ("RL_hip_joint", 0.1),
    ("RR_hip_joint", -0.1),
    ("FL_thigh_joint", 0.8),
    ("FR_thigh_joint", 0.8),
    ("RL_thigh_joint", 1.0),
    ("RR_thigh_joint", 1.0),
    ("FL_calf_joint", -1.5),
    ("FR_calf_joint", -1.5),
    ("RL_calf_joint", -1.5),
    ("RR_calf_joint", -1.5),
]
ACTION_SCALE = 0.25
KP, KD, TAU_LIMIT = 25.0, 0.5, 23.5
OBS_SCALES = dict(lin_vel=2.0, ang_vel=0.25, joint_vel=0.05)
GOAL_POS_TOL, GOAL_Z_TOL, GOAL_YAW_TOL = 0.35, 0.12, 0.45
HEIGHTMAP_DIM, HEIGHTMAP_RES = 21, 0.2  # 4 m x 4 m grid
FOOT_XS, FOOT_YS = (-0.15, -0.05, 0.05, 0.15), (-0.05, 0.05)  # 0.3 x 0.1 m, 0.1 m res


def quat_rotate_inv(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a world vector into the body frame of quaternion q (w, x, y, z)."""
    w, x, y, z = q_wxyz
    # rows are the body axes in world coordinates, i.e. the transpose of body->world
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
        [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
    ])
    return rot @ v


def yaw_of(q_wxyz: np.ndarray) -> float:
    w, x, y, z = q_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def build_scene_xml(go2_xml: str, step_height: float, flat: bool = False) -> str:
    """Straight staircase (12 steps x 0.25 m tread + 1.5 m landing) like the training terrain.

    With ``flat=True`` the staircase is omitted (the goal then sits on the ground ahead) to
    sanity-check the observation/action pipeline without stair interaction.
    """
    n, tread, width, landing = 12, 0.25, 1.7, 1.5
    x0 = 1.0  # bottom of the stairs
    top = n * step_height
    lines = [
        '<mujoco model="go2 stair demo">',
        f'  <include file="{os.path.basename(go2_xml)}"/>',
        '  <statistic center="2.5 0 0.6" extent="3.0"/>',
        '  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>',
        '    <rgba haze="0.15 0.25 0.35 1"/><global azimuth="-130" elevation="-20"/></visual>',
        '  <asset><texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>',
        '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"',
        '      markrgb="0.8 0.8 0.8" width="300" height="300"/>',
        '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/></asset>',
        '  <worldbody>',
        '    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>',
        '    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>',
    ]
    if not flat:
        for k in range(n):
            sz = (k + 1) * step_height
            cx = x0 + (k + 0.5) * tread
            lines.append(
                f'    <geom name="step{k}" type="box" pos="{cx:.4f} 0 {sz / 2:.4f}" size="{tread / 2:.4f} {width / 2:.4f} {sz / 2:.4f}"/>'
            )
        lines.append(
            f'    <geom name="landing" type="box" pos="{x0 + n * tread + landing / 2:.4f} 0 {top / 2:.4f}" '
            f'size="{landing / 2:.4f} {width / 2:.4f} {top / 2:.4f}"/>'
        )
    lines += ["  </worldbody>", "</mujoco>"]

    path = os.path.join(os.path.dirname(go2_xml), "_stair_demo_generated.xml")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


class StairDemo:
    def __init__(self, scene_xml: str, onnx_path: str, step_height: float, flat: bool = False):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.model.opt.timestep = 0.005  # training: 4 x 0.005 s per 50 Hz control step
        # the shipped go2.xml uses frictionless contacts (condim=1) and low friction;
        # align with the training terrain (sliding friction 1.0, 3D friction cone)
        self.model.geom_condim[:] = 3
        self.model.geom_friction[:, :] = [1.0, 0.02, 0.01]
        self.data = mujoco.MjData(self.model)
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        m = self.model
        self.base_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        free_jnt = m.body_jntadr[self.base_body]
        self.qpos0, self.qvel0 = m.jnt_qposadr[free_jnt], m.jnt_dofadr[free_jnt]
        self.foot_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
        gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.gyro_adr = m.sensor_adr[gyro_id]

        # joint name -> (qpos index, dof index) and actuator -> joint name
        self.joint_idx = {}
        asset_joint_names = {name for name, _ in ASSET_JOINTS}
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name in asset_joint_names:
                self.joint_idx[name] = (m.jnt_qposadr[j], m.jnt_dofadr[j])
        self.act_joint = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0]) for a in range(m.nu)]

        # ray geom groups: terrain (worldbody) = 0, robot = 1
        self.geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        for g in range(m.ngeom):
            m.geom_group[g] = 0 if m.geom_bodyid[g] == 0 else 1

        # staircase geometry
        self.n_steps, self.tread, self.step_height = 12, 0.25, step_height
        self.x0 = 1.0
        self.top = 0.0 if flat else self.n_steps * step_height
        self.spawn = np.array([self.x0 - 0.5, 0.0])
        self.goal_xy = np.array([self.x0 + self.n_steps * self.tread - 0.1, 0.0])
        self.goal_yaw = 0.0
        self.stand_z = 0.40  # measured after settling
        self.height_bias = 0.0  # matches the height map to the training distribution
        self.foot_bias = np.zeros(4)  # per-foot height bias matching the Isaac foot frames

    # ------------------------------------------------------------------ helpers
    def reset(self):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.qpos0: self.qpos0 + 7] = [*self.spawn, 0.40, 1.0, 0.0, 0.0, 0.0]
        for name, default in ASSET_JOINTS:
            self.data.qpos[self.joint_idx[name][0]] = default
        mujoco.mj_forward(self.model, self.data)

    def hold_default(self, seconds: float):
        target = {name: default for name, default in ASSET_JOINTS}
        for _ in range(int(round(seconds / 0.02))):
            self.apply_pd(target, n_sub=4)
        self.stand_z = float(self.data.xpos[self.base_body][2])
        # Isaac reference: foot heights at spawn are ~0.0998 m (front) and ~0.1090 m (rear)
        isaac_foot_ref = np.array([0.0998, 0.0998, 0.1090, 0.1090])  # FL, FR, RL, RR
        foot_z = np.array([float(self.data.xpos[b][2]) for b in self.foot_bodies])
        self.foot_bias = isaac_foot_ref - foot_z

    def apply_pd(self, target: dict, n_sub: int):
        m, d = self.model, self.data
        kp, kd, lim = KP, KD, TAU_LIMIT
        for _ in range(n_sub):
            for a, jname in enumerate(self.act_joint):
                qadr, vadr = self.joint_idx[jname]
                tau = kp * (target[jname] - d.qpos[qadr]) - kd * d.qvel[vadr]
                d.ctrl[a] = float(np.clip(tau, -lim, lim))
            mujoco.mj_step(m, d)

    def ray_height(self, x: float, y: float, z_top: float) -> float:
        dist = mujoco.mj_ray(self.model, self.data, np.array([x, y, z_top]), np.array([0.0, 0.0, -1.0]),
                             self.geomgroup, 1, -1, None)
        if dist < 0:
            return z_top + 5.0  # nothing below: treat as a far miss (clips to -1)
        return z_top - dist

    def build_obs(self, last_action: np.ndarray) -> np.ndarray:
        m, d = self.model, self.data
        root_pos = d.qpos[self.qpos0: self.qpos0 + 3].copy()
        root_quat = d.qpos[self.qpos0 + 3: self.qpos0 + 7].copy()
        lin_vel_b = quat_rotate_inv(root_quat, d.qvel[self.qvel0: self.qvel0 + 3].copy())
        ang_vel_b = d.sensordata[self.gyro_adr: self.gyro_adr + 3].copy()
        gravity_b = quat_rotate_inv(root_quat, np.array([0.0, 0.0, -1.0]))
        joint_pos = np.array([d.qpos[self.joint_idx[name][0]] - default for name, default in ASSET_JOINTS])
        joint_vel = np.array([d.qvel[self.joint_idx[name][1]] for name, _ in ASSET_JOINTS])

        goal_w = np.array([self.goal_xy[0], self.goal_xy[1], self.top + self.stand_z])
        rel_pos = quat_rotate_inv(root_quat, goal_w - root_pos)
        rel_heading = (self.goal_yaw - yaw_of(root_quat) + math.pi) % (2 * math.pi) - math.pi
        goal = np.concatenate([rel_pos, [rel_heading]])

        yaw = yaw_of(root_quat)
        c, s = math.cos(yaw), math.sin(yaw)

        def world_xy(dx: float, dy: float) -> tuple[float, float]:
            return root_pos[0] + c * dx - s * dy, root_pos[1] + s * dx + c * dy

        # the training env's base frame sits ~0.08 m higher above the ground than this MJCF,
        # so the base-frame-relative height map gets a bias; the foot bodies also sit lower
        # than in the Isaac model, so foot heights get their own per-foot bias
        bias = self.height_bias
        foot = []
        for fidx, body in enumerate(self.foot_bodies):
            fx, fy, fz = d.xpos[body]
            for dy in FOOT_YS:
                for dx in FOOT_XS:
                    wx, wy = fx + (c * dx - s * dy), fy + (s * dx + c * dy)
                    foot.append(np.clip(fz + self.foot_bias[fidx] - self.ray_height(wx, wy, fz), -1.0, 1.0))
        foot = np.array(foot)

        half = HEIGHTMAP_DIM // 2
        coords = np.linspace(-half * HEIGHTMAP_RES, half * HEIGHTMAP_RES, HEIGHTMAP_DIM)
        heightmap = np.empty(HEIGHTMAP_DIM * HEIGHTMAP_DIM, dtype=np.float64)
        idx = 0
        # Isaac flattens the grid with x fastest: index = y * 21 + x
        for dy in coords:
            for dx in coords:
                wx, wy = world_xy(dx, dy)
                heightmap[idx] = np.clip(
                    root_pos[2] + bias - self.ray_height(wx, wy, root_pos[2] + 5.0), -1.0, 1.0
                )
                idx += 1

        obs = np.concatenate([
            lin_vel_b * OBS_SCALES["lin_vel"],
            ang_vel_b * OBS_SCALES["ang_vel"],
            gravity_b,
            joint_pos,
            joint_vel * OBS_SCALES["joint_vel"],
            last_action,
            goal,
            foot,
            heightmap,
        ])
        return obs.astype(np.float32)

    def reached(self) -> bool:
        d = self.data
        pos = d.qpos[self.qpos0: self.qpos0 + 3]
        quat = d.qpos[self.qpos0 + 3: self.qpos0 + 7]
        d_xy = float(np.linalg.norm(self.goal_xy - pos[:2]))
        d_z = abs(self.top + self.stand_z - pos[2])
        yaw_err = abs((self.goal_yaw - yaw_of(quat) + math.pi) % (2 * math.pi) - math.pi)
        return d_xy < GOAL_POS_TOL and d_z < GOAL_Z_TOL and yaw_err < GOAL_YAW_TOL


def main():
    parser = argparse.ArgumentParser(description="Replay the stair policy in MuJoCo.")
    parser.add_argument("--onnx", default="deploy/stair/policy.onnx", help="Path to the exported ONNX policy.")
    parser.add_argument("--go2_xml", default="stack/unitree_mujoco/unitree_robots/go2/go2.xml", help="Go2 MJCF path.")
    parser.add_argument("--step_height", type=float, default=0.08, help="Stair step height (m).")
    parser.add_argument("--duration", type=float, default=30.0, help="Rollout duration (s).")
    parser.add_argument("--headless", action="store_true", default=False, help="No viewer; print progress.")
    parser.add_argument("--debug", action="store_true", default=False, help="Print observation/action diagnostics.")
    parser.add_argument(
        "--height_bias",
        type=float,
        default=None,
        help="Offset added to the foot-height/height-map observations (default: auto, "
        "0.40 - settled base height, matching the Isaac base-frame height).",
    )
    parser.add_argument("--flat", action="store_true", default=False, help="No staircase (pipeline sanity check).")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    go2_xml = os.path.join(root, args.go2_xml)
    onnx_path = os.path.join(root, args.onnx)
    scene_xml = build_scene_xml(go2_xml, args.step_height, flat=args.flat)
    demo = StairDemo(scene_xml, onnx_path, args.step_height, flat=args.flat)

    demo.reset()
    demo.hold_default(0.5)
    demo.height_bias = (0.40 - demo.stand_z) if args.height_bias is None else args.height_bias
    print(f"stand height after settling: {demo.stand_z:.3f} m, height-obs bias: {demo.height_bias:+.3f} m")

    viewer = None
    if not args.headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(demo.model, demo.data)
    last_action = np.zeros(12)
    reached_at = None
    n_ctrl = int(round(args.duration / 0.02))
    try:
        for step in range(n_ctrl):
            obs = demo.build_obs(last_action)
            action = demo.session.run(None, {demo.input_name: obs[None, :]})[0][0]
            if args.debug and step % 25 == 0:
                print(
                    f"[dbg t={step * 0.02:4.1f}] obs[:6]={np.round(obs[:6], 2).tolist()} "
                    f"joint_pos[:4]={np.round(obs[9:13], 2).tolist()} goal={np.round(obs[45:49], 2).tolist()} "
                    f"foot[0:4]={np.round(obs[49:53], 2).tolist()} hmap[min,max]=({obs[81:].min():.2f},{obs[81:].max():.2f}) "
                    f"action|max|={np.abs(action).max():.2f} std={action.std():.2f}"
                )
            target = {name: default + ACTION_SCALE * float(action[i]) for i, (name, default) in enumerate(ASSET_JOINTS)}
            demo.apply_pd(target, n_sub=4)
            last_action = action
            if reached_at is None and demo.reached():
                reached_at = step * 0.02
                print(f"[OK] goal reached at t = {reached_at:.2f} s")
            if step % 50 == 0:
                pos = demo.data.qpos[demo.qpos0: demo.qpos0 + 3]
                d_xy = float(np.linalg.norm(demo.goal_xy - pos[:2]))
                print(f"t={step * 0.02:5.1f}s  base=({pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:4.2f})  dist_xy={d_xy:4.2f}")
            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    break
    finally:
        if viewer is not None:
            viewer.close()

    pos = demo.data.qpos[demo.qpos0: demo.qpos0 + 3]
    print(f"final base: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), stair top z = {demo.top:.2f}")
    print(f"result: {'REACHED' if reached_at is not None else 'NOT reached'} (limit {GOAL_POS_TOL} m xy / {GOAL_Z_TOL} m z)")


if __name__ == "__main__":
    main()