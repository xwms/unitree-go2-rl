"""Reset events for the stair tasks."""

from __future__ import annotations

import math

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_angle_axis, sample_uniform

from .stair_cache import get_stair_cache

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

Z_AXIS = torch.tensor([0.0, 0.0, 1.0])


def reset_root_state_from_stair(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lateral_range: tuple[float, float] = (-0.3, 0.3),
    yaw_range: tuple[float, float] = (-math.pi / 4, math.pi / 4),
    descend_prob: float = 0.5,
):
    """Reset the robot root state at the terrain-relative spawn pose.

    Stage 2 samples the travel direction per episode: ascending spawns at the stair bottom
    facing up, descending spawns on the top landing facing down. Stage 1 places the robot
    at the pyramid platform center with a uniform ``yaw_range`` of ``(-pi, pi)``.
    """
    cache = get_stair_cache(env)
    # per-episode direction (pyramid envs never descend); stored for the command / rewards
    descend = (torch.rand(len(env_ids), device=env.device) < descend_prob) & cache["is_stair"][env_ids]
    mode = getattr(env, "_stair_descend", None)
    if mode is None or mode.shape[0] != env.num_envs:
        mode = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._stair_descend = mode
    mode[env_ids] = descend

    root_states = env.scene[asset_cfg.name].data.default_root_state[env_ids].clone()
    env_origins = env.scene.env_origins[env_ids]
    offsets = torch.where(descend.unsqueeze(1), cache["spawn_down_offsets"][env_ids], cache["spawn_offsets"][env_ids])
    root_states[:, :3] += env_origins + offsets
    root_states[:, 0] += sample_uniform(*lateral_range, (len(env_ids),), device=env.device)
    root_states[:, 1] += sample_uniform(*lateral_range, (len(env_ids),), device=env.device)
    base_yaw = torch.where(descend, cache["spawn_yaws_down"][env_ids], cache["spawn_yaws"][env_ids])
    yaw = base_yaw + sample_uniform(*yaw_range, (len(env_ids),), device=env.device)
    root_states[:, 3:7] = quat_from_angle_axis(yaw, Z_AXIS.repeat(len(env_ids), 1).to(env.device))
    env.scene[asset_cfg.name].write_root_state_to_sim(root_states, env_ids)
