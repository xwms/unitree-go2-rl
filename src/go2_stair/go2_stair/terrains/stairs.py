"""Programmatic stair terrains (straight / L-shaped / spiral) with centerline metadata.

These terrains replicate the stage-2 stair environments of arXiv 2602.03087v2
(adaptive stair climbing/descending for the Unitree Go2). The stair geometry is
generated per difficulty level and, for each generated terrain, the centerline of
the staircase is recorded so that reward terms (center / path rewards), goal
sampling and spawn placement can query it at runtime.

All mesh functions follow the Isaac Lab convention: geometry is built in the
``(0, 0) - (size[0], size[1])`` frame and ``TerrainGenerator._get_terrain_mesh``
re-centers it afterwards. The recorded metadata is expressed in the same final,
tile-centered frame (shifted by ``-size/2`` along x and y).

The Isaac Lab ``TerrainImporter`` instantiates the generator from
``TerrainGeneratorCfg.class_type`` but does not keep a reference to it, so the
per-terrain metadata is stored in a module-level registry keyed by the generator
cfg object (see :func:`get_stair_metadata`).
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGenerator, TerrainGeneratorCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPyramidStairsTerrainCfg
from isaaclab.utils import configclass

# -----------------------------------------------------------------------------
# Metadata registry (TerrainImporter keeps no reference to the generator)
# -----------------------------------------------------------------------------

_STAIR_META_REGISTRY: dict[int, dict] = {}


def get_stair_metadata(terrain_generator_cfg: TerrainGeneratorCfg) -> dict | None:
    """Return the per-terrain metadata recorded for the given generator cfg, if any."""
    return _STAIR_META_REGISTRY.get(id(terrain_generator_cfg))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _lerp(difficulty: float, value_range: tuple[float, float]) -> float:
    return value_range[0] + difficulty * (value_range[1] - value_range[0])


def _box(dims: tuple[float, float, float], pos: tuple[float, float, float], yaw: float = 0.0) -> trimesh.Trimesh:
    box = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))
    if yaw != 0.0:
        box.apply_transform(trimesh.transformations.rotation_matrix(yaw, [0, 0, 1]))
    return box


def _ground_slab(size: tuple[float, float]) -> trimesh.Trimesh:
    return _box((size[0], size[1], 0.1), (size[0] / 2.0, size[1] / 2.0, -0.05))


def _shift_to_tile_center(points: np.ndarray, size: tuple[float, float]) -> np.ndarray:
    """Apply the same ``-size/2`` x/y shift that ``TerrainGenerator._get_terrain_mesh`` applies."""
    shifted = np.array(points, dtype=np.float64)
    shifted[..., 0] -= size[0] / 2.0
    shifted[..., 1] -= size[1] / 2.0
    return shifted


# -----------------------------------------------------------------------------
# Terrain mesh functions (frame: (0,0) - (size[0], size[1]), re-centered by Isaac Lab)
# -----------------------------------------------------------------------------


def straight_stairs_terrain(difficulty: float, cfg: "StraightStairsTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Straight staircase: ``num_steps`` steps along +x, centered in the tile."""
    sh = _lerp(difficulty, cfg.step_height_range)
    w = _lerp(difficulty, cfg.width_range)
    n = cfg.num_steps
    length = n * cfg.step_depth
    x0 = (cfg.size[0] - length) / 2.0  # bottom of the stairs
    y0 = cfg.size[1] / 2.0

    meshes = [_ground_slab(cfg.size)]
    for k in range(n):
        top = (k + 1) * sh
        meshes.append(_box((cfg.step_depth, w, top), (x0 + (k + 0.5) * cfg.step_depth, y0, top / 2.0)))
    origin = np.array([x0, y0, 0.0])  # stair bottom, ground level
    return meshes, origin


def l_stairs_terrain(difficulty: float, cfg: "LStairsTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """L-shaped staircase: segment 1 along +x, then a 90 deg turn into segment 2 along +y."""
    sh = _lerp(difficulty, cfg.step_height_range)
    w = _lerp(difficulty, cfg.width_range)
    n1 = cfg.num_steps_seg1
    n2 = max(0, int(round(_lerp(difficulty, cfg.seg2_length_range) / cfg.step_depth)))
    d = cfg.step_depth
    x0 = (cfg.size[0] - n1 * d) / 2.0
    y0 = (cfg.size[1] - n2 * d) / 2.0
    corner_x = x0 + n1 * d

    meshes = [_ground_slab(cfg.size)]
    # segment 1 along +x
    for k in range(n1):
        top = (k + 1) * sh
        meshes.append(_box((d, w, top), (x0 + (k + 0.5) * d, y0, top / 2.0)))
    # segment 2 along +y, rising from the top of segment 1
    for j in range(n2):
        top = (n1 + j + 1) * sh
        meshes.append(_box((w, d, top), (corner_x, y0 + (j + 0.5) * d, top / 2.0)))
    origin = np.array([x0, y0, 0.0])
    return meshes, origin


def spiral_stairs_terrain(difficulty: float, cfg: "SpiralStairsTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Spiral staircase: steps arranged along a circular arc centered at the tile center."""
    sh = _lerp(difficulty, cfg.step_height_range)
    w = _lerp(difficulty, cfg.width_range)
    turns = _lerp(difficulty, cfg.turns_range)
    r_mid = cfg.inner_radius + w / 2.0
    dalpha = cfg.step_depth / r_mid  # angular step so the tread depth is constant along the centerline
    n_steps = max(1, int(round(turns * 2.0 * np.pi / dalpha)))
    cx, cy = cfg.size[0] / 2.0, cfg.size[1] / 2.0

    meshes = [_ground_slab(cfg.size)]
    for k in range(n_steps):
        alpha = (k + 0.5) * dalpha
        top = (k + 1) * sh
        meshes.append(
            _box(
                (cfg.step_depth, w, top),
                (cx + r_mid * np.cos(alpha), cy + r_mid * np.sin(alpha), top / 2.0),
                yaw=alpha,
            )
        )
    # origin: on the ground, just before the first step (which starts at angle 0)
    origin = np.array([cx + r_mid, cy, 0.0])
    return meshes, origin


# -----------------------------------------------------------------------------
# Sub-terrain configs
# -----------------------------------------------------------------------------


@configclass
class StairSubTerrainCfg(SubTerrainBaseCfg):
    """Common parameters for the procedural stair terrains."""

    step_height_range: tuple[float, float] = (0.02, 0.12)
    """Step height range interpolated by difficulty (in m)."""

    width_range: tuple[float, float] = (2.0, 1.4)
    """Staircase width range interpolated by difficulty (in m)."""

    step_depth: float = 0.25
    """Depth of each step / tread length along the centerline (in m)."""

    def compute_metadata(self, difficulty: float) -> dict:
        raise NotImplementedError


@configclass
class StraightStairsTerrainCfg(StairSubTerrainCfg):
    """Configuration for a straight staircase terrain."""

    function = straight_stairs_terrain

    num_steps: int = 12
    """Number of steps."""

    def compute_metadata(self, difficulty: float) -> dict:
        sh = _lerp(difficulty, self.step_height_range)
        w = _lerp(difficulty, self.width_range)
        n = self.num_steps
        length = n * self.step_depth
        x0 = (self.size[0] - length) / 2.0
        y0 = self.size[1] / 2.0
        # centerline: front edge of each step
        centerline = np.stack(
            [np.linspace(x0, x0 + length, n + 1), np.full(n + 1, y0), sh * np.arange(n + 1)], axis=-1
        )
        spawn_pos = np.array([x0 - 0.5, y0, 0.0])
        return {
            "kind": "straight",
            "step_height": sh,
            "stair_width": w,
            "centerline": _shift_to_tile_center(centerline, self.size),
            "s_end": float(length),
            "spawn_pos": _shift_to_tile_center(spawn_pos, self.size),
            "spawn_offset": np.array([-0.5, 0.0, 0.0]),
            "dir": np.array([1.0, 0.0]),
            "dir_end": np.array([1.0, 0.0]),
            "platform_z": 0.0,
        }


@configclass
class LStairsTerrainCfg(StairSubTerrainCfg):
    """Configuration for an L-shaped staircase terrain."""

    function = l_stairs_terrain

    num_steps_seg1: int = 8
    """Number of steps in the first (approach) segment."""

    seg2_length_range: tuple[float, float] = (0.0, 3.0)
    """Length of the second segment (after the turn) interpolated by difficulty (in m)."""

    def compute_metadata(self, difficulty: float) -> dict:
        sh = _lerp(difficulty, self.step_height_range)
        w = _lerp(difficulty, self.width_range)
        n1 = self.num_steps_seg1
        n2 = max(0, int(round(_lerp(difficulty, self.seg2_length_range) / self.step_depth)))
        d = self.step_depth
        x0 = (self.size[0] - n1 * d) / 2.0
        y0 = (self.size[1] - n2 * d) / 2.0
        corner_x = x0 + n1 * d
        # segment 1
        pts1 = np.stack([np.linspace(x0, corner_x, n1 + 1), np.full(n1 + 1, y0), sh * np.arange(n1 + 1)], axis=-1)
        # segment 2 (continues past the corner along +y)
        pts2 = np.stack(
            [np.full(n2 + 1, corner_x), y0 + d * np.arange(n2 + 1), sh * (n1 + np.arange(n2 + 1))], axis=-1
        )
        centerline = np.concatenate([pts1, pts2[1:]], axis=0)
        spawn_pos = np.array([x0 - 0.5, y0, 0.0])
        return {
            "kind": "L",
            "step_height": sh,
            "stair_width": w,
            "centerline": _shift_to_tile_center(centerline, self.size),
            "s_end": float((n1 + n2) * d),
            "spawn_pos": _shift_to_tile_center(spawn_pos, self.size),
            "spawn_offset": np.array([-0.5, 0.0, 0.0]),
            "dir": np.array([1.0, 0.0]),
            "dir_end": np.array([0.0, 1.0]),
            "platform_z": 0.0,
        }


@configclass
class SpiralStairsTerrainCfg(StairSubTerrainCfg):
    """Configuration for a spiral staircase terrain."""

    function = spiral_stairs_terrain

    inner_radius: float = 0.4
    """Inner radius of the spiral (in m)."""

    turns_range: tuple[float, float] = (0.2, 0.5)
    """Number of turns interpolated by difficulty."""

    def compute_metadata(self, difficulty: float) -> dict:
        sh = _lerp(difficulty, self.step_height_range)
        w = _lerp(difficulty, self.width_range)
        turns = _lerp(difficulty, self.turns_range)
        r_mid = self.inner_radius + w / 2.0
        dalpha = self.step_depth / r_mid
        n_steps = max(1, int(round(turns * 2.0 * np.pi / dalpha)))
        cx, cy = self.size[0] / 2.0, self.size[1] / 2.0
        alphas = dalpha * np.arange(n_steps + 1)
        centerline = np.stack(
            [cx + r_mid * np.cos(alphas), cy + r_mid * np.sin(alphas), sh * np.arange(n_steps + 1)], axis=-1
        )
        # spawn 0.5 m before the first step (which starts at angle 0, tangent along +y)
        spawn_pos = np.array([cx + r_mid, cy - 0.5, 0.0])
        alpha_end = n_steps * dalpha
        return {
            "kind": "spiral",
            "step_height": sh,
            "stair_width": w,
            "centerline": _shift_to_tile_center(centerline, self.size),
            "s_end": float(n_steps * self.step_depth),
            "spawn_pos": _shift_to_tile_center(spawn_pos, self.size),
            "spawn_offset": np.array([0.0, -0.5, 0.0]),
            "dir": np.array([0.0, 1.0]),
            "dir_end": np.array([-np.sin(alpha_end), np.cos(alpha_end)]),
            "platform_z": 0.0,
        }


@configclass
class PyramidStairsWithMetaCfg(MeshPyramidStairsTerrainCfg):
    """Built-in pyramid stairs terrain with platform metadata for stage-1 goal sampling."""

    goal_radius: float = 2.2
    """Radius around the platform center within which goals are sampled (in m)."""

    def compute_metadata(self, difficulty: float) -> dict:
        step_height = _lerp(difficulty, self.step_height_range)
        half = (self.size[0] - 2.0 * self.border_width) / 2.0
        num_steps_x = (self.size[0] - 2 * self.border_width - self.platform_width) // (2 * self.step_width) + 1
        num_steps_y = (self.size[1] - 2 * self.border_width - self.platform_width) // (2 * self.step_width) + 1
        num_steps = int(min(num_steps_x, num_steps_y))
        platform_z = (num_steps + 1) * step_height  # matches the origin of pyramid_stairs_terrain
        return {
            "kind": "pyramid",
            "step_height": step_height,
            "stair_width": 0.0,
            "centerline": np.zeros((1, 3)),
            "s_end": 0.0,
            "spawn_pos": np.zeros(3),
            "spawn_offset": np.zeros(3),
            "dir": np.array([1.0, 0.0]),
            "dir_end": np.array([1.0, 0.0]),
            "platform_z": platform_z,
            # pyramid goal-z helper: z = (min(num_steps, floor((half - cheb)/step_width)) + 1) * step_height
            "pyramid_half": half,
            "pyramid_step_width": self.step_width,
            "pyramid_num_steps": num_steps,
            "goal_radius": self.goal_radius,
        }


# -----------------------------------------------------------------------------
# Terrain generator
# -----------------------------------------------------------------------------


@configclass
class StairTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Configuration for the stair terrain generator (curriculum mode only)."""

    size: tuple[float, float] = (8.0, 8.0)
    """Size of each terrain tile (in m)."""

    curriculum: bool = True
    """Whether to use the curriculum mode. Always True for the stair tasks."""

    border_width: float = 20.0
    """Width of the surrounding border (in m)."""

    num_rows: int = 10
    """Number of difficulty levels."""

    num_cols: int = 3
    """Number of sub-terrain columns."""

    difficulty_range: tuple[float, float] = (0.0, 1.0)
    """Difficulty range mapped onto the rows."""

    use_cache: bool = False

    def __post_init__(self):
        if not self.curriculum:
            raise ValueError("StairTerrainGeneratorCfg requires curriculum=True (terrains are indexed by level).")
        # late binding: the generator class is defined below
        self.class_type = StairTerrainGenerator


class StairTerrainGenerator(TerrainGenerator):
    """Terrain generator that records per-terrain metadata (centerline, spawn, params).

    The Isaac Lab importer does not keep a reference to the generator instance, so the
    metadata is published into the module-level registry keyed by the generator cfg.
    """

    def __init__(self, cfg: StairTerrainGeneratorCfg, device: str = "cpu"):
        self._stair_meta: dict[tuple[int, int], dict] = {}
        super().__init__(cfg, device)
        _STAIR_META_REGISTRY[id(cfg)] = self._stair_meta

    def _generate_curriculum_terrains(self):
        """Mirror of :meth:`TerrainGenerator._generate_curriculum_terrains` that records metadata.

        Iteration order is column-major; the metadata is keyed by (row, col).
        """
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        # find the sub-terrain index for each column
        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # curriculum-based sub-terrains
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                lower, upper = self.cfg.difficulty_range
                difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty
                # generate terrain
                sub_cfg = sub_terrains_cfgs[sub_indices[sub_col]]
                mesh, origin = self._get_terrain_mesh(difficulty, sub_cfg)
                # add to sub-terrains
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_cfg)
                # record metadata for the stair sub-terrains
                if isinstance(sub_cfg, (StairSubTerrainCfg, PyramidStairsWithMetaCfg)):
                    self._stair_meta[(sub_row, sub_col)] = sub_cfg.compute_metadata(difficulty)
