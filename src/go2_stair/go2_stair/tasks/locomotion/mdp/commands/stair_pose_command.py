"""Goal pose command sampled on the staircase centerlines / pyramid platform."""

from __future__ import annotations

import math

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

from ..stair_cache import get_stair_cache

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class StairPoseCommandCfg(CommandTermCfg):
    """Configuration for the stair goal pose command."""

    class_type: type = MISSING

    goal_mode: str = "stairs"
    """How goals are sampled: 'stairs' (end of the staircase) or 'pyramid' (platform disk)."""

    goal_radius: float = 2.2
    """Radius of the goal sampling disk in pyramid mode (in m)."""

    resampling_time_range: tuple[float, float] = (20.0, 20.0)
    """Time between resamplings; the env cfg overrides this with the episode length."""

    def __post_init__(self):
        # late binding: the command class is defined below
        self.class_type = StairPoseCommand


class StairPoseCommand(CommandTerm):
    """Goal pose command for the stair tasks.

    Stage 1 (``goal_mode="pyramid"``): the goal position is sampled uniformly on a disk
    around the pyramid platform center, at the local step/platform height; the goal heading
    is uniform. Stage 2 (``goal_mode="stairs"``): the goal is the top of the staircase with
    the heading aligned to the final segment direction.
    """

    cfg: StairPoseCommandCfg

    def __init__(self, cfg: StairPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator."""
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        # goal buffers (world frame)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        # metrics
        self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "StairPoseCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tGoal mode: {self.cfg.goal_mode}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}"
        )

    @property
    def command(self) -> torch.Tensor:
        """The desired base pose (position + heading) in the world frame. Shape: (num_envs, 4)."""
        return torch.cat((self.pos_command_w, self.heading_command_w.unsqueeze(-1)), dim=1)

    def _resample_command(self, env_ids: torch.Tensor):
        """Sample a new goal pose for the given environments."""
        env_ids = env_ids.to(self.device)
        # keep the command for the whole episode
        self.time_left[env_ids] = float("inf")
        cache = get_stair_cache(self._env)

        if self.cfg.goal_mode == "stairs":
            stair_envs = env_ids[cache["is_stair"][env_ids]]
            pyr_envs = env_ids[~cache["is_stair"][env_ids]]
        else:
            stair_envs = env_ids[:0]
            pyr_envs = env_ids

        # goal at the top of the staircase
        if len(stair_envs) > 0:
            self.pos_command_w[stair_envs] = cache["goal_pos"][stair_envs]
            dirs = cache["goal_dir"][stair_envs]
            self.heading_command_w[stair_envs] = torch.atan2(dirs[:, 1], dirs[:, 0])

        # goal on the pyramid platform disk
        if len(pyr_envs) > 0:
            r = torch.sqrt(torch.rand(len(pyr_envs), device=self.device)) * cache["goal_radius"][pyr_envs]
            ang = torch.rand(len(pyr_envs), device=self.device) * 2.0 * math.pi
            dx = r * torch.cos(ang)
            dy = r * torch.sin(ang)
            self.pos_command_w[pyr_envs, 0] = self._env.scene.env_origins[pyr_envs, 0] + dx
            self.pos_command_w[pyr_envs, 1] = self._env.scene.env_origins[pyr_envs, 1] + dy
            # local pyramid surface height for the sampled point
            cheb = torch.maximum(dx.abs(), dy.abs())
            k = torch.floor((cache["pyramid_half"][pyr_envs] - cheb) / cache["pyramid_step_width"][pyr_envs]).clamp(
                min=0.0
            )
            z_surface = (torch.minimum(cache["pyramid_num_steps"][pyr_envs].float(), k) + 1.0) * cache[
                "pyramid_step_height"
            ][pyr_envs]
            self.pos_command_w[pyr_envs, 2] = z_surface + cache["robot_default_z"]
            self.heading_command_w[pyr_envs] = (
                torch.rand(len(pyr_envs), device=self.device) * 2.0 * math.pi - math.pi
            )

    def _update_command(self):
        """Commands are stored in the world frame; nothing to update."""

    def _update_metrics(self):
        """Compute the 2-D position and heading errors toward the goal."""
        root_pos = self.robot.data.root_pos_w
        self.metrics["error_pos_2d"][:] = torch.norm(self.pos_command_w[:, :2] - root_pos[:, :2], dim=1)
        self.metrics["error_heading"][:] = torch.abs(
            wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)
        )
