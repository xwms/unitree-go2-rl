"""Reset events for the stair tasks."""

from __future__ import annotations

import math

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import ArticulationCfg
from isaaclab.utils.math import quat_from_angle_axis, torch_rand_float

from .stair_cache import get_stair_cache

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

Z_AXIS = torch.tensor([0.0, 0.0, 1.0])


def reset_root_state_from_stair(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: ArticulationCfg = MISSING,
    lateral_range: tuple[float, float] = (-0.3, 0.3),
    yaw_range: tuple[float, float] = (-math.pi / 4, math.pi / 4),
):
    """Reset the robot root state at the terrain-relative spawn pose.

    The robot is placed at the stair spawn offset (stage 2) or the pyramid platform center
    (stage 1), with lateral noise and a heading noise around the approach direction. For
    pyramid envs a uniform ``yaw_range`` of ``(-pi, pi)`` yields fully random headings.
    """
    if asset_cfg == MISSING:
        asset_cfg = env.scene["robot"].cfg
    cache = get_stair_cache(env)
    root_states = env.scene[asset_cfg.name].data.default_root_state.clone()
    env_origins = env.scene.env_origins[env_ids]
    root_states[:, :3] += env_origins + cache["spawn_offsets"][env_ids]
    root_states[:, 0] += torch_rand_float(*lateral_range, (len(env_ids), 1), device=env.device)
    root_states[:, 1] += torch_rand_float(*lateral_range, (len(env_ids), 1), device=env.device)
    yaw = cache["spawn_yaws"][env_ids] + torch_rand_float(*yaw_range, (len(env_ids),), device=env.device)
    root_states[:, 3:7] = quat_from_angle_axis(yaw, Z_AXIS.repeat(len(env_ids), 1).to(env.device))
    env.scene[asset_cfg.name].write_root_state_to_sim(root_states, env_ids)
