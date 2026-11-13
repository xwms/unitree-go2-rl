"""Curriculum and termination terms for the stair tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stair_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "goal_command",
    success_tolerance: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Terrain curriculum: level up when the goal is reached, down when the robot falls.

    Goal reached: 3-D position error below ``success_tolerance``. Fallen: episode terminated
    for a reason other than time-out (and not by reaching the goal).
    """
    cmd = env.command_manager.get_term(command_name)
    pos_error = torch.norm(cmd.pos_command_w[env_ids] - env.scene["robot"].data.root_pos_w[env_ids], dim=1)
    # reset_terminated/reset_time_outs only exist after the first env step; on the very
    # first reset the commands have not been resampled yet, so no curriculum change
    if getattr(env, "reset_terminated", None) is None:
        return torch.zeros_like(pos_error, dtype=torch.bool), torch.zeros_like(pos_error, dtype=torch.bool)
    move_up = pos_error < success_tolerance
    move_down = env.reset_terminated[env_ids] & ~env.reset_time_outs[env_ids]
    move_down = move_down & ~move_up
    return move_up, move_down


def goal_reached(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_command",
    pos_tolerance: float = 0.35,
    z_tolerance: float = 0.35,
    yaw_tolerance: float = 0.45,
) -> torch.Tensor:
    """Terminate the episode when the goal pose is reached within the tolerances."""
    cmd = env.command_manager.get_term(command_name)
    root_pos = env.scene["robot"].data.root_pos_w
    d_xy = torch.norm(cmd.pos_command_w[:, :2] - root_pos[:, :2], dim=1)
    d_z = torch.abs(cmd.pos_command_w[:, 2] - root_pos[:, 2])
    yaw_err = torch.abs(wrap_to_pi(cmd.heading_command_w - env.scene["robot"].data.heading_w))
    return (d_xy < pos_tolerance) & (d_z < z_tolerance) & (yaw_err < yaw_tolerance)
