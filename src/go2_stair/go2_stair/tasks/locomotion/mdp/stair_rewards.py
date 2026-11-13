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
            "prev_progress": torch.zeros(env.num_envs, device=env.device),
            "prev_touch": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        }
        env._stair_rw_buffers = bufs
    return bufs


# -----------------------------------------------------------------------------
# Navigation rewards (stage 1 and stage 2)
# -----------------------------------------------------------------------------


def nav_progress_reward(env: ManagerBasedRLEnv, command_name: str = "goal_command", sigma: float = 5.0) -> torch.Tensor:
    """Navigation reward ``r_nav = 1 - tanh(d / sigma)`` with ``d`` the 3-D distance to the goal."""
    cmd = env.command_manager.get_term(command_name)
    d = torch.norm(cmd.pos_command_w - env.scene["robot"].data.root_pos_w, dim=1)
    return 1.0 - torch.tanh(d / sigma)


def heading_error_penalty(env: ManagerBasedRLEnv, command_name: str = "goal_command") -> torch.Tensor:
    """Penalty proportional to the absolute heading error toward the goal heading."""
    cmd = env.command_manager.get_term(command_name)
    error = torch.abs(wrap_to_pi(cmd.heading_command_w - env.scene["robot"].data.heading_w))
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
    env: ManagerBasedRLEnv, dt: float | None = None, sigma: float = 2.0, max_progress: float = 1.5
) -> torch.Tensor:
    """Path reward: arc-length progress along the centerline, modulated by the center term."""
    dt = env.step_dt if dt is None else dt
    cache = get_stair_cache(env)
    bufs = _stair_buffers(env)
    arc, lat = project_to_centerline(env, cache, env.scene["robot"].data.root_pos_w[:, :2])
    progress = torch.clip((arc - bufs["prev_progress"]) / dt, min=0.0, max=max_progress)
    center = torch.exp(-(lat**2) / (2.0 * sigma**2))
    bufs["prev_progress"][:] = arc
    bufs["prev_progress"][env.reset_terminated] = 0.0
    return progress * center


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


def leg_contact_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalty (1.0) while any body in the given group is in contact.

    The paper penalizes thigh contacts with -8.0 and shank/foot contacts with -0.2; the
    weights are set through separate reward terms with different ``body_names`` filters.
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    return (torch.norm(force, dim=-1).max(dim=1).values > threshold).float()


def flying_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, air_time_threshold: float = 0.1
) -> torch.Tensor:
    """Penalty while all feet are airborne for longer than the threshold (returns 1.0)."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    foot_air_time = sensor.data.air_time[:, sensor_cfg.body_ids]
    return (foot_air_time.min(dim=1).values > air_time_threshold).float()


def step_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Sparse reward (1.0) on a foot touchdown event."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    force = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    touched = (torch.norm(force, dim=-1) > threshold).any(dim=1)
    bufs = _stair_buffers(env)
    reward = (touched & ~bufs["prev_touch"]).float()
    bufs["prev_touch"][:] = touched
    bufs["prev_touch"][env.reset_terminated] = False
    return reward
