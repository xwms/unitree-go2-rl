"""World-frame stair geometry cache derived from the terrain generator metadata.

The curriculum manager moves environment origins when difficulty levels change, and the
terrain metadata is keyed by ``(terrain_level, terrain_type)``. This module builds a
per-environment cache (centerlines, arc lengths, goal/spawn poses) and rebuilds it lazily
whenever the terrain assignment of any environment changes.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from go2_stair.terrains import get_stair_metadata

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_stair_cache(env: ManagerBasedRLEnv) -> dict:
    """Return the stair geometry cache, rebuilding it if the terrain assignment changed."""
    terrain = env.scene.terrain
    levels = terrain.terrain_levels
    types = terrain.terrain_types
    cache = getattr(env, "_stair_meta_cache", None)
    if cache is None or not torch.equal(cache["levels"], levels) or not torch.equal(cache["types"], types):
        cache = _build_stair_cache(env, levels, types)
        env._stair_meta_cache = cache
    return cache


def _build_stair_cache(env: ManagerBasedRLEnv, levels: torch.Tensor, types: torch.Tensor) -> dict:
    """Build the per-environment geometry cache from the terrain generator metadata."""
    meta_all = get_stair_metadata(env.scene.terrain.cfg.terrain_generator)
    if meta_all is None:
        raise RuntimeError(
            "Stair task requires StairTerrainGenerator as terrain generator "
            "(metadata registry returned nothing for the current terrain_generator cfg)."
        )
    num_envs = env.num_envs
    device = env.device
    origins = env.scene.env_origins
    robot_default_z = float(env.scene["robot"].data.default_root_state[0, 2])

    # per-environment tensors
    tile_ids = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    is_stair = torch.zeros(num_envs, dtype=torch.bool, device=device)
    goal_pos = torch.zeros(num_envs, 3, device=device)
    goal_dir = torch.zeros(num_envs, 2, device=device)
    spawn_offsets = torch.zeros(num_envs, 3, device=device)
    spawn_yaws = torch.zeros(num_envs, device=device)
    platform_z = torch.zeros(num_envs, device=device)
    goal_radius = torch.zeros(num_envs, device=device)
    pyramid_half = torch.zeros(num_envs, device=device)
    pyramid_step_width = torch.zeros(num_envs, device=device)
    pyramid_num_steps = torch.zeros(num_envs, device=device)
    pyramid_step_height = torch.zeros(num_envs, device=device)

    tiles: dict[tuple[int, int], int] = {}
    tile_geoms: list[dict | None] = []

    for e in range(num_envs):
        key = (int(levels[e]), int(types[e]))
        if key not in tiles:
            meta = meta_all[key]
            tiles[key] = len(tile_geoms)
            o = origins[e]
            if meta["kind"] == "pyramid":
                tile_geoms.append(None)
            else:
                centerline = torch.as_tensor(meta["centerline"], dtype=torch.float32, device=device)
                centerline[:, :2] += o[:2]
                centerline[:, 2] += o[2]
                seg_len = torch.norm(centerline[1:, :2] - centerline[:-1, :2], dim=1)
                cum_arc = torch.cat([torch.zeros(1, device=device), torch.cumsum(seg_len, dim=0)])
                tile_geoms.append(
                    {
                        "centerline": centerline,
                        "cum_arc": cum_arc,
                        "dir_end": torch.as_tensor(meta["dir_end"], dtype=torch.float32, device=device),
                    }
                )
        tile_id = tiles[key]
        tile_ids[e] = tile_id
        meta = meta_all[key]
        if meta["kind"] == "pyramid":
            platform_z[e] = meta["platform_z"]
            goal_radius[e] = meta["goal_radius"]
            pyramid_half[e] = meta["pyramid_half"]
            pyramid_step_width[e] = meta["pyramid_step_width"]
            pyramid_num_steps[e] = meta["pyramid_num_steps"]
            pyramid_step_height[e] = meta["step_height"]
        else:
            is_stair[e] = True
            geom = tile_geoms[tile_id]
            dir_end = geom["dir_end"]
            goal_pos[e, :2] = geom["centerline"][-1, :2] - 0.1 * dir_end
            goal_pos[e, 2] = geom["centerline"][-1, 2] + robot_default_z
            goal_dir[e] = dir_end
            dir0 = torch.as_tensor(meta["dir"], dtype=torch.float32, device=device)
            spawn_yaws[e] = torch.atan2(dir0[1], dir0[0])
        spawn_offsets[e] = torch.as_tensor(meta["spawn_offset"], dtype=torch.float32, device=device)

    return {
        "levels": levels.clone(),
        "types": types.clone(),
        "tile_ids": tile_ids,
        "is_stair": is_stair,
        "goal_pos": goal_pos,
        "goal_dir": goal_dir,
        "spawn_offsets": spawn_offsets,
        "spawn_yaws": spawn_yaws,
        "platform_z": platform_z,
        "goal_radius": goal_radius,
        "pyramid_half": pyramid_half,
        "pyramid_step_width": pyramid_step_width,
        "pyramid_num_steps": pyramid_num_steps,
        "pyramid_step_height": pyramid_step_height,
        "robot_default_z": robot_default_z,
        "tiles": tile_geoms,
    }


def project_to_centerline(env: ManagerBasedRLEnv, cache: dict, pos_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project positions onto the stair centerlines.

    Returns:
        arc: Arc-length progress along the centerline (0 for pyramid envs).
        lat: Signed lateral distance to the centerline (0 for pyramid envs).
    """
    num_envs = pos_xy.shape[0]
    arc = torch.zeros(num_envs, device=env.device)
    lat = torch.zeros(num_envs, device=env.device)
    for tile_id, geom in enumerate(cache["tiles"]):
        if geom is None:
            continue
        env_ids = (cache["tile_ids"] == tile_id).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            continue
        cl = geom["centerline"][:, :2]
        cum = geom["cum_arc"]
        a, b = cl[:-1], cl[1:]
        seg = b - a
        seg_len2 = torch.sum(seg * seg, dim=1).clamp(min=1e-8)
        ap = pos_xy[env_ids].unsqueeze(1) - a.unsqueeze(0)
        t = torch.sum(ap * seg.unsqueeze(0), dim=-1) / seg_len2
        t = t.clamp(0.0, 1.0)
        proj = a.unsqueeze(0) + t.unsqueeze(-1) * seg.unsqueeze(0)
        dist2 = torch.sum((pos_xy[env_ids].unsqueeze(1) - proj) ** 2, dim=-1)
        seg_idx = dist2.argmin(dim=1)
        t_sel = t.gather(1, seg_idx.unsqueeze(1)).squeeze(1)
        arc[env_ids] = cum[seg_idx] + t_sel * (cum[seg_idx + 1] - cum[seg_idx])
        s = seg[seg_idx]
        ap_sel = pos_xy[env_ids] - a[seg_idx]
        cross = ap_sel[:, 0] * s[:, 1] - ap_sel[:, 1] * s[:, 0]
        lat[env_ids] = torch.sign(cross) * torch.sqrt(dist2.gather(1, seg_idx.unsqueeze(1)).squeeze(1))
    return arc, lat
