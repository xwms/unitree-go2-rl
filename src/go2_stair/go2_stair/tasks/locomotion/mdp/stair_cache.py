"""World-frame stair geometry cache derived from the terrain generator metadata.

The curriculum manager moves environment origins when difficulty levels change, and the
terrain metadata is keyed by ``(terrain_level, terrain_type)``. This module builds a
static GPU lookup table once per terrain generator configuration and gathers per-environment
geometry and centerline projections purely on GPU without host-device synchronization.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from go2_stair.terrains import get_stair_metadata

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["get_stair_cache", "project_to_centerline"]


def _build_stair_tables(env: ManagerBasedRLEnv) -> dict:
    """Build static per-tile GPU lookup tables once from the terrain generator metadata."""
    gen_cfg = env.scene.terrain.cfg.terrain_generator
    meta_all = get_stair_metadata(gen_cfg)
    if meta_all is None:
        raise RuntimeError(
            "Stair task requires StairTerrainGenerator as terrain generator "
            "(metadata registry returned nothing for the current terrain_generator cfg)."
        )
    device = torch.device(env.device)
    robot_default_z = float(env.scene["robot"].data.default_root_state[0, 2])

    num_rows = max(k[0] for k in meta_all.keys()) + 1
    num_cols = max(k[1] for k in meta_all.keys()) + 1
    num_tiles = num_rows * num_cols

    is_stair_tbl = torch.zeros(num_tiles, dtype=torch.bool)
    goal_end_local_tbl = torch.zeros((num_tiles, 3), dtype=torch.float32)
    goal_dir_tbl = torch.zeros((num_tiles, 2), dtype=torch.float32)
    spawn_offset_tbl = torch.zeros((num_tiles, 3), dtype=torch.float32)
    spawn_yaw_tbl = torch.zeros(num_tiles, dtype=torch.float32)
    platform_z_tbl = torch.zeros(num_tiles, dtype=torch.float32)
    goal_radius_tbl = torch.zeros(num_tiles, dtype=torch.float32)
    pyr_half_tbl = torch.zeros(num_tiles, dtype=torch.float32)
    pyr_step_w_tbl = torch.zeros(num_tiles, dtype=torch.float32)
    pyr_num_steps_tbl = torch.zeros(num_tiles, dtype=torch.float32)
    pyr_step_h_tbl = torch.zeros(num_tiles, dtype=torch.float32)

    max_segs = 1
    for meta in meta_all.values():
        if meta["kind"] != "pyramid":
            cl = np.asarray(meta["centerline"], dtype=np.float32)
            max_segs = max(max_segs, max(1, cl.shape[0] - 1))

    seg_a_local = torch.zeros((num_tiles, max_segs, 2), dtype=torch.float32)
    seg_b_local = torch.zeros((num_tiles, max_segs, 2), dtype=torch.float32)
    cum_arc = torch.zeros((num_tiles, max_segs + 1), dtype=torch.float32)
    seg_valid = torch.zeros((num_tiles, max_segs), dtype=torch.bool)
    has_any_stair = False

    for (r, c), meta in meta_all.items():
        tid = r * num_cols + c
        spawn_offset_tbl[tid] = torch.as_tensor(meta["spawn_offset"], dtype=torch.float32)
        if meta["kind"] == "pyramid":
            platform_z_tbl[tid] = float(meta["platform_z"])
            goal_radius_tbl[tid] = float(meta["goal_radius"])
            pyr_half_tbl[tid] = float(meta["pyramid_half"])
            pyr_step_w_tbl[tid] = float(meta["pyramid_step_width"])
            pyr_num_steps_tbl[tid] = float(meta["pyramid_num_steps"])
            pyr_step_h_tbl[tid] = float(meta["step_height"])
            seg_valid[tid, 0] = True
        else:
            has_any_stair = True
            is_stair_tbl[tid] = True
            cl = torch.as_tensor(meta["centerline"], dtype=torch.float32)
            dir_end = torch.as_tensor(meta["dir_end"], dtype=torch.float32)
            dir0 = torch.as_tensor(meta["dir"], dtype=torch.float32)
            goal_end_local_tbl[tid] = cl[-1]
            goal_dir_tbl[tid] = dir_end
            spawn_yaw_tbl[tid] = torch.atan2(dir0[1], dir0[0])

            n_seg = cl.shape[0] - 1
            a = cl[:-1, :2]
            b = cl[1:, :2]
            sl = torch.norm(b - a, dim=1)
            seg_a_local[tid, :n_seg] = a
            seg_b_local[tid, :n_seg] = b
            cum_arc[tid, 1 : n_seg + 1] = torch.cumsum(sl, dim=0)
            seg_valid[tid, :n_seg] = True

    return {
        "cfg_id": id(gen_cfg),
        "device": device,
        "num_cols": num_cols,
        "has_any_stair": has_any_stair,
        "robot_default_z": robot_default_z,
        "is_stair_tbl": is_stair_tbl.to(device),
        "goal_end_local_tbl": goal_end_local_tbl.to(device),
        "goal_dir_tbl": goal_dir_tbl.to(device),
        "spawn_offset_tbl": spawn_offset_tbl.to(device),
        "spawn_yaw_tbl": spawn_yaw_tbl.to(device),
        "platform_z_tbl": platform_z_tbl.to(device),
        "goal_radius_tbl": goal_radius_tbl.to(device),
        "pyr_half_tbl": pyr_half_tbl.to(device),
        "pyr_step_w_tbl": pyr_step_w_tbl.to(device),
        "pyr_num_steps_tbl": pyr_num_steps_tbl.to(device),
        "pyr_step_h_tbl": pyr_step_h_tbl.to(device),
        "seg_a_local": seg_a_local.to(device),
        "seg_b_local": seg_b_local.to(device),
        "cum_arc": cum_arc.to(device),
        "seg_valid": seg_valid.to(device),
    }


def get_stair_cache(env: ManagerBasedRLEnv) -> dict:
    """Return per-environment stair geometry tensors via pure-GPU table lookup."""
    terrain = env.scene.terrain
    levels = terrain.terrain_levels
    types = terrain.terrain_types
    origins = env.scene.env_origins
    gen_cfg = terrain.cfg.terrain_generator

    tbl = getattr(env, "_stair_static_tables", None)
    if tbl is None or tbl["cfg_id"] != id(gen_cfg) or tbl["device"] != torch.device(env.device):
        tbl = _build_stair_tables(env)
        env._stair_static_tables = tbl

    tile_ids = levels.long() * tbl["num_cols"] + types.long()
    is_stair = tbl["is_stair_tbl"][tile_ids]
    goal_dir = tbl["goal_dir_tbl"][tile_ids]
    end_w = tbl["goal_end_local_tbl"][tile_ids] + origins
    goal_xy = end_w[:, :2] - 0.1 * goal_dir
    goal_z = (end_w[:, 2] + tbl["robot_default_z"]).unsqueeze(1)
    goal_pos = torch.where(
        is_stair.unsqueeze(1),
        torch.cat((goal_xy, goal_z), dim=1),
        torch.zeros_like(origins),
    )

    return {
        "_tbl": tbl,
        "levels": levels,
        "types": types,
        "origins": origins,
        "tile_ids": tile_ids,
        "is_stair": is_stair,
        "goal_pos": goal_pos,
        "goal_dir": goal_dir,
        "spawn_offsets": tbl["spawn_offset_tbl"][tile_ids],
        "spawn_yaws": tbl["spawn_yaw_tbl"][tile_ids],
        "platform_z": tbl["platform_z_tbl"][tile_ids],
        "goal_radius": tbl["goal_radius_tbl"][tile_ids],
        "pyramid_half": tbl["pyr_half_tbl"][tile_ids],
        "pyramid_step_width": tbl["pyr_step_w_tbl"][tile_ids],
        "pyramid_num_steps": tbl["pyr_num_steps_tbl"][tile_ids],
        "pyramid_step_height": tbl["pyr_step_h_tbl"][tile_ids],
        "robot_default_z": tbl["robot_default_z"],
    }


def project_to_centerline(env: ManagerBasedRLEnv, cache: dict, pos_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project positions onto the stair centerlines without host-device synchronization.

    Returns:
        arc: Arc-length progress along the centerline (0 for pyramid envs).
        lat: Signed lateral distance to the centerline (0 for pyramid envs).
    """
    tbl = cache["_tbl"]
    num_envs = pos_xy.shape[0]
    if not tbl["has_any_stair"]:
        zero = torch.zeros(num_envs, dtype=pos_xy.dtype, device=pos_xy.device)
        return zero, zero.clone()

    tids = cache["tile_ids"]
    origins_xy = cache["origins"][:, :2].unsqueeze(1)
    a = tbl["seg_a_local"][tids] + origins_xy
    b = tbl["seg_b_local"][tids] + origins_xy
    seg = b - a
    seg_len2 = torch.sum(seg * seg, dim=-1).clamp(min=1e-8)
    valid = tbl["seg_valid"][tids]
    cum = tbl["cum_arc"][tids]

    ap = pos_xy.unsqueeze(1) - a
    t = (torch.sum(ap * seg, dim=-1) / seg_len2).clamp(0.0, 1.0)
    proj = a + t.unsqueeze(-1) * seg
    dist2 = torch.sum((pos_xy.unsqueeze(1) - proj) ** 2, dim=-1)
    dist2_masked = dist2.masked_fill(~valid, float("inf"))

    seg_idx = dist2_masked.argmin(dim=1, keepdim=True)
    t_sel = t.gather(1, seg_idx).squeeze(1)
    c0 = cum.gather(1, seg_idx).squeeze(1)
    c1 = cum.gather(1, seg_idx + 1).squeeze(1)
    arc_raw = c0 + t_sel * (c1 - c0)

    idx2 = seg_idx.unsqueeze(-1).expand(-1, 1, 2)
    s_sel = seg.gather(1, idx2).squeeze(1)
    ap_sel = ap.gather(1, idx2).squeeze(1)
    cross = ap_sel[:, 0] * s_sel[:, 1] - ap_sel[:, 1] * s_sel[:, 0]
    min_dist2 = dist2.gather(1, seg_idx).squeeze(1)
    lat_raw = torch.sign(cross) * torch.sqrt(min_dist2)

    is_st = cache["is_stair"]
    zero = torch.zeros_like(arc_raw)
    return torch.where(is_st, arc_raw, zero), torch.where(is_st, lat_raw, zero)
