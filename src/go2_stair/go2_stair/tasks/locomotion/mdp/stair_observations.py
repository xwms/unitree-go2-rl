"""Observation terms for the stair tasks: goal pose, foot heights, height map."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_pose_rel(env: ManagerBasedRLEnv, command_name: str = "goal_command") -> torch.Tensor:
    """Goal pose relative to the base frame: (dx, dy, dz, heading error). Shape: (num_envs, 4)."""
    cmd = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    rel_pos = quat_apply_inverse(robot.data.root_quat_w, cmd.pos_command_w - robot.data.root_pos_w)
    rel_heading = wrap_to_pi(cmd.heading_command_w - robot.data.heading_w)
    return torch.cat([rel_pos, rel_heading.unsqueeze(-1)], dim=-1)


def foot_heights_observation(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...] = ("FL_foot_scanner", "FR_foot_scanner", "RL_foot_scanner", "RR_foot_scanner"),
    clip_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Height of each foot above the terrain surface, from the per-foot ray casters.

    Each foot scanner fires an 8-ray grid; the 32 values are concatenated in the order of
    ``sensor_names``. Shape: (num_envs, 32).
    """
    parts = []
    for name in sensor_names:
        sensor = env.scene[name]
        origin_z = sensor.data.pos_w[:, 2].unsqueeze(-1)
        hits_z = sensor.data.ray_hits_w[..., 2]
        parts.append(origin_z - hits_z)
    return torch.clip(torch.cat(parts, dim=-1), min=clip_range[0], max=clip_range[1])


def height_map_observation(
    env: ManagerBasedRLEnv,
    sensor_name: str = "height_scanner",
    clip_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Height map around the robot: root height above the terrain at each grid ray.

    The grid has 21 x 21 rays (441 values) arranged row-major; the policy CNN reshapes it.
    Shape: (num_envs, 441).
    """
    sensor = env.scene[sensor_name]
    root_z = env.scene["robot"].data.root_pos_w[:, 2].unsqueeze(-1)
    hits_z = sensor.data.ray_hits_w[..., 2]
    return torch.clip(root_z - hits_z, min=clip_range[0], max=clip_range[1])
