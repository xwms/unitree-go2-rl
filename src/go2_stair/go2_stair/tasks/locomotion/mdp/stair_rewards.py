"""Reward terms for the stair climbing/descending tasks (arXiv 2602.03087v2)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi

from .stair_cache import get_stair_cache, project_to_centerline

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _stair_buffers(env: ManagerBasedRLEnv) -> dict:
    """Lazily create per-env history buffers shared by the reward terms."""
    bufs = getattr(env, "_stair_rw_buffers", None)
    if bufs is None:
        bufs = {
            "best_arc": torch.zeros(env.num_envs, device=env.device),
            "topped": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "prev_touch": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "foot_contact": torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device),
        }
        env._stair_rw_buffers = bufs
    return bufs


# -----------------------------------------------------------------------------
# Navigation rewards (stage 1 and stage 2)
# -----------------------------------------------------------------------------


def _stair_height_gate(env: ManagerBasedRLEnv, cache: dict, z_consistency: float) -> torch.Tensor:
    """True where the robot base height matches the stair surface at its centerline projection.

    The centerline projection is 2-D, so a robot on the flat ground beside or behind the
    staircase projects onto the stairs and would otherwise collect the nav/path rewards
    without climbing. Non-stair (pyramid) envs always pass.
    """
    robot = env.scene["robot"]
    arc, _ = project_to_centerline(env, cache, robot.data.root_pos_w[:, :2])
    top = cache["goal_pos"][:, 2] - cache["origins"][:, 2] - cache["robot_default_z"]
    z_surf = cache["origins"][:, 2] + (arc / cache["s_end"].clamp(min=1e-6)) * top
    z_robot = robot.data.root_pos_w[:, 2]
    on_stair = torch.abs(z_robot - (z_surf + cache["robot_default_z"])) < z_consistency
    return torch.where(cache["is_stair"], on_stair, True)


def nav_progress_reward(
    env: ManagerBasedRLEnv, command_name: str = "goal_command", sigma: float = 5.0, z_consistency: float | None = None
) -> torch.Tensor:
    """Navigation reward ``r_nav = 1 - tanh(d / sigma)`` with ``d`` the 3-D distance to the goal.

    With ``z_consistency`` set, the reward only pays while the robot is at the stair
    surface height of its projection (see :func:`_stair_height_gate`): the goal sits
    inside the top step, so the raw 3-D distance can be minimized from the flat ground
    beside or behind the staircase without ever climbing.
    """
    cmd = env.command_manager.get_term(command_name)
    d = torch.norm(cmd.pos_command_w - env.scene["robot"].data.root_pos_w, dim=1)
    reward = 1.0 - torch.tanh(d / sigma)
    if z_consistency is None:
        return reward
    return reward * _stair_height_gate(env, get_stair_cache(env), z_consistency)


def heading_error_penalty(
    env: ManagerBasedRLEnv, command_name: str = "goal_command", near_radius: float | None = None
) -> torch.Tensor:
    """Penalty proportional to the absolute heading error toward the goal heading.

    If ``near_radius`` is set, the penalty is gated to the final approach: it only
    applies within that 2-D distance of the goal so the approach heading is shaped
    without constraining the robot during the climb itself.
    """
    cmd = env.command_manager.get_term(command_name)
    error = torch.abs(wrap_to_pi(cmd.heading_command_w - env.scene["robot"].data.heading_w))
    if near_radius is not None:
        d_xy = torch.norm(cmd.pos_command_w[:, :2] - env.scene["robot"].data.root_pos_w[:, :2], dim=1)
        error = error * (d_xy < near_radius)
    return error


# -----------------------------------------------------------------------------
# Centerline rewards (stage 2)
# -----------------------------------------------------------------------------


def centerline_reward(env: ManagerBasedRLEnv, sigma: float = 0.5) -> torch.Tensor:
    """Center reward ``r_center = exp(-d_lat^2 / (2 sigma^2))`` with lateral distance to the centerline."""
    cache = get_stair_cache(env)
    _, lat = project_to_centerline(env, cache, env.scene["robot"].data.root_pos_w[:, :2])
    return torch.exp(-(lat**2) / (2.0 * sigma**2))


def path_progress_reward(
    env: ManagerBasedRLEnv,
    dt: float | None = None,
    sigma: float = 2.0,
    max_progress: float = 1.5,
    z_consistency: float | None = None,
) -> torch.Tensor:
    """Path reward: arc-length progress along the centerline in the travel direction.

    Progress is measured against the best signed progress reached so far in the episode
    (ascending counts the arc up, descending counts it down): backing up and re-covering
    the same stretch cannot farm the reward. With ``z_consistency`` set, the reward only
    pays while the robot is at the stair surface height of its projection, so a ground-level
    lap alongside the staircase (whose xy projection tracks the centerline to the end)
    cannot farm the progress.
    """
    dt = env.step_dt if dt is None else dt
    cache = get_stair_cache(env)
    bufs = _stair_buffers(env)
    arc, lat = project_to_centerline(env, cache, env.scene["robot"].data.root_pos_w[:, :2])
    signed_arc = arc * torch.where(cache["descend"], -1.0, 1.0)
    progress = torch.clip((signed_arc - bufs["best_arc"]) / dt, min=0.0, max=max_progress)
    center = torch.exp(-(lat**2) / (2.0 * sigma**2))
    reward = progress * center
    done = env.reset_terminated | env.reset_time_outs
    bufs["best_arc"] = torch.maximum(bufs["best_arc"], signed_arc)
    bufs["best_arc"][done] = 0.0
    if z_consistency is None:
        return reward
    return reward * _stair_height_gate(env, cache, z_consistency)


def goal_topped_bonus(env: ManagerBasedRLEnv, margin: float = 0.3, z_tolerance: float = 0.13) -> torch.Tensor:
    """One-shot bonus when the robot first climbs onto the final approach of the staircase.

    Fires once per episode when the arc-length projection enters the last ``margin``
    meters of the centerline and the robot is near the goal height. The height gate
    excludes the flat ground just past the top edge, which projects onto the centerline
    endpoint (full arc length) without any climbing; ``z_tolerance`` must stay below the
    minimum stair top height (0.14 m at spiral level 0).
    """
    cache = get_stair_cache(env)
    bufs = _stair_buffers(env)
    robot = env.scene["robot"]
    arc, _ = project_to_centerline(env, cache, robot.data.root_pos_w[:, :2])
    d_z = torch.abs(cache["goal_pos"][:, 2] - robot.data.root_pos_w[:, 2])
    done = env.reset_terminated | env.reset_time_outs
    crossed = (arc >= cache["s_end"] - margin) & (d_z < z_tolerance) & cache["is_stair"] & ~cache["descend"]
    first = crossed & ~bufs["topped"] & ~done
    bufs["topped"] |= crossed
    bufs["topped"][done] = False
    return first.float()


def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_command",
    pos_tolerance: float = 0.35,
    z_tolerance: float = 0.12,
    yaw_tolerance: float = 0.45,
) -> torch.Tensor:
    """Success payoff on the step the goal condition is met (mirrors the ``goal_reached`` termination).

    Terminating on success pays nothing by itself, while hovering at the goal doorstep
    collects the dense nav rewards indefinitely, so without this bonus the policy learns
    to never finish. Return units are ``weight * step_dt``; the weight must exceed the
    discounted hover value of the nav terms, i.e. ``raw_per_step * step_dt / (1 - gamma)``.
    """
    cmd = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    d_xy = torch.norm(cmd.pos_command_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    d_z = torch.abs(cmd.pos_command_w[:, 2] - robot.data.root_pos_w[:, 2])
    yaw_err = torch.abs(wrap_to_pi(cmd.heading_command_w - robot.data.heading_w))
    return ((d_xy < pos_tolerance) & (d_z < z_tolerance) & (yaw_err < yaw_tolerance)).float()


# -----------------------------------------------------------------------------
# Regularization rewards
# -----------------------------------------------------------------------------


def power_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Mechanical power penalty ``sum(|torque * velocity|)`` over the joints."""
    robot = env.scene["robot"]
    return torch.abs(robot.data.applied_torque * robot.data.joint_vel).sum(dim=1)


def stationary_power_penalty(env: ManagerBasedRLEnv, speed_threshold: float = 0.05) -> torch.Tensor:
    """Power penalty applied only while the base is nearly stationary."""
    robot = env.scene["robot"]
    power = torch.abs(robot.data.applied_torque * robot.data.joint_vel).sum(dim=1)
    stationary = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1) < speed_threshold
    return power * stationary


def action_rate_penalty(env: ManagerBasedRLEnv, dt: float | None = None) -> torch.Tensor:
    """Penalty on the squared action change rate."""
    dt = env.step_dt if dt is None else dt
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1) / dt


def joint_position_limit_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty proportional to the total joint position limit violation."""
    robot = env.scene["robot"]
    limits = robot.data.soft_joint_pos_limits
    pos = robot.data.joint_pos
    violation = torch.clip(limits[..., 0] - pos, min=0.0) + torch.clip(pos - limits[..., 1], min=0.0)
    return violation.sum(dim=1)


def joint_velocity_limit_penalty(env: ManagerBasedRLEnv, soft_ratio: float = 1.0) -> torch.Tensor:
    """Penalty proportional to the total joint velocity limit violation.

    The 2.3 API exposes soft joint velocity limits as a magnitude per joint
    (``soft_joint_vel_limits``), not a (lower, upper) pair.
    """
    robot = env.scene["robot"]
    out_of_limits = robot.data.joint_vel.abs() - robot.data.soft_joint_vel_limits * soft_ratio
    return torch.clip(out_of_limits, min=0.0, max=1.0).sum(dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty on the squared joint velocities."""
    return torch.sum(torch.square(env.scene["robot"].data.joint_vel), dim=1)


def joint_acceleration_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty on the squared joint accelerations."""
    robot = env.scene["robot"]
    return torch.sum(torch.square(robot.data.joint_acc), dim=1)


def base_roll_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared sine of the base roll; unlike flat-orientation penalties it leaves pitch free."""
    return torch.square(env.scene["robot"].data.projected_gravity_b[:, 1])


def leg_contact_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalty (1.0) while any body in the given group is in contact.

    The paper penalizes thigh contacts with -8.0 and shank/foot contacts with -0.2; the
    weights are set through separate reward terms with different ``body_names`` filters.
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    return (torch.norm(force, dim=-1).max(dim=1).values.max(dim=1).values > threshold).float()


def flying_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, air_time_threshold: float = 0.1
) -> torch.Tensor:
    """Penalty while all feet are airborne for longer than the threshold (returns 1.0)."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    foot_air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return (foot_air_time.min(dim=1).values > air_time_threshold).float()


def step_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Sparse reward (1.0) on a foot touchdown event."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    force = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    touched = (torch.norm(force, dim=-1) > threshold).any(dim=1)
    bufs = _stair_buffers(env)
    reward = (touched & ~bufs["prev_touch"]).float()
    done = env.reset_terminated | env.reset_time_outs
    bufs["prev_touch"][:] = touched
    bufs["prev_touch"][done] = False
    return reward


# -----------------------------------------------------------------------------
# Gait shaping (stage 2)
# -----------------------------------------------------------------------------


def feet_air_time_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.2,
    max_credit: float = 0.4,
) -> torch.Tensor:
    """Credit the swing time of each foot when it lands, for swings longer than ``threshold``.

    Encourages deliberate steps (lift the foot, swing, place it) instead of a fast shuffle:
    the credit is ``clip(air_time - threshold, 0, max_credit)`` per landing foot.
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    foot_air = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    contact = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1) > 1.0
    bufs = _stair_buffers(env)
    first = contact & ~bufs["foot_contact"]
    credit = torch.clip(foot_air - threshold, min=0.0, max=max_credit)
    reward = (credit * first.float()).sum(dim=1)
    done = env.reset_terminated | env.reset_time_outs
    bufs["foot_contact"][:] = contact
    bufs["foot_contact"][done] = False
    return reward


def feet_slide_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sum of horizontal foot speeds while a foot is in contact (penalizes skating)."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    contact = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1) > 1.0
    foot_vel = torch.norm(robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    return (foot_vel * contact.float()).sum(dim=1)


def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant per-step cost: prices the episode time so sprinting stops being free."""
    return torch.ones(env.num_envs, device=env.device)


def feet_trot_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Clock-free trot preference on the foot contact states.

    Rewards diagonal feet agreeing in contact (FL~RR, FR~RL) and penalizes the front /
    rear pairs agreeing (a bound/"legs-together" gait). All four feet loaded (standing)
    scores 0, a diagonal trot scores +2, a bound or pace-like pattern scores negative.
    Unlike a phase-clock reward it needs no observable clock, which a feed-forward policy
    cannot synchronize to.
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    contact = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1) > 1.0
    legs = {}
    for i, bid in enumerate(sensor_cfg.body_ids):
        name = sensor.body_names[bid]
        legs[name[0] + name[1]] = i  # FL / FR / RL / RR
    fl, fr, rl, rr = (contact[:, legs[k]] for k in ("FL", "FR", "RL", "RR"))
    diagonal = (fl == rr).float() + (fr == rl).float()
    pairs = (fl == fr).float() + (rl == rr).float()
    return diagonal - pairs


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Variance of the last swing / stance times across the feet (encourages a symmetric gait)."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    last_air = torch.clip(sensor.data.last_air_time[:, sensor_cfg.body_ids], max=0.5)
    last_contact = torch.clip(sensor.data.last_contact_time[:, sensor_cfg.body_ids], max=0.5)
    return torch.var(last_air, dim=1) + torch.var(last_contact, dim=1)


def feet_swing_lift_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_drop: float = 0.25,
    std: float = 0.05,
) -> torch.Tensor:
    """Reward swinging feet that tuck up to ~``target_drop`` below the base.

    A terrain-agnostic clearance term (the official absolute-height version only works on
    flat ground): the foot gets ``exp(-(drop - target_drop)^2 / 2std^2)`` while it is in
    the air, so visible lifts are worth more than dragging the feet across the steps.
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    contact = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1) > 1.0
    drop = robot.data.root_pos_w[:, 2].unsqueeze(1) - robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    per_foot = torch.exp(-torch.square(drop - target_drop) / (2.0 * std**2))
    return (per_foot * (~contact).float()).sum(dim=1)


def hind_swing_drop_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    target_drop: float = 0.25,
    max_excess: float = 0.15,
) -> torch.Tensor:
    """Penalize hind swing feet that hang more than ``target_drop`` below the base."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    hind_ids = getattr(env, "_stair_hind_ids", None)
    if hind_ids is None:
        foot_names = [sensor.body_names[i] for i in sensor_cfg.body_ids]
        hind_ids = (foot_names.index("RL_foot"), foot_names.index("RR_foot"))
        env._stair_hind_ids = hind_ids
    contact = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1) > 1.0
    drop = robot.data.root_pos_w[:, 2].unsqueeze(1) - robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    excess = (drop[:, hind_ids] - target_drop).clamp(min=0.0, max=max_excess)
    return (excess * (~contact[:, hind_ids]).float()).sum(dim=1)
