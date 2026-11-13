"""Two-stage stair climbing/descending environments for the Unitree Go2 (arXiv 2602.03087v2).

Stage 1: built-in pyramid stairs terrain with navigation rewards (goal reaching on the
platform). Stage 2: straight / L-shaped / spiral stair terrains with centerline and path
rewards. Both stages share the same 522-dim observation (45 proprio + 4 goal + 32 foot
heights + 441 height map), so stage 2 resumes from the stage 1 checkpoint.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from go2_stair.tasks.locomotion import mdp
from go2_stair.terrains import (
    LStairsTerrainCfg,
    PyramidStairsWithMetaCfg,
    SpiralStairsTerrainCfg,
    StairTerrainGeneratorCfg,
    StraightStairsTerrainCfg,
)

# -----------------------------------------------------------------------------
# Terrain generators
# -----------------------------------------------------------------------------


def stage1_terrain_cfg() -> StairTerrainGeneratorCfg:
    """Stage 1: pyramid stairs terrain (Isaac Lab built-in), 10 difficulty levels."""
    return StairTerrainGeneratorCfg(
        num_cols=1,
        sub_terrains={
            "pyramid_stairs": PyramidStairsWithMetaCfg(
                proportion=1.0,
                step_height_range=(0.02, 0.12),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                goal_radius=2.2,
            ),
        },
    )


def stage2_terrain_cfg() -> StairTerrainGeneratorCfg:
    """Stage 2: straight / L-shaped / spiral stair terrains, one column each."""
    stair_common = dict(
        step_height_range=(0.02, 0.12),
        width_range=(2.0, 1.4),
        step_depth=0.25,
    )
    return StairTerrainGeneratorCfg(
        num_cols=3,
        sub_terrains={
            "straight": StraightStairsTerrainCfg(proportion=1.0, num_steps=12, **stair_common),
            "L": LStairsTerrainCfg(
                proportion=1.0, num_steps_seg1=8, seg2_length_range=(0.0, 3.0), **stair_common
            ),
            "spiral": SpiralStairsTerrainCfg(
                proportion=1.0, inner_radius=0.4, turns_range=(0.2, 0.5), **stair_common
            ),
        },
    )


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------


@configclass
class StairSceneCfg(InteractiveSceneCfg):
    """Scene with terrain, Go2 robot, height map / foot ray casters and contact sensors."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=stage2_terrain_cfg(),
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 21 x 21 = 441 rays around the robot (4 m x 4 m grid at 0.2 m resolution)
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[4.0, 4.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # 4 x 2 = 8 rays under each foot (0.3 m x 0.1 m grid at 0.1 m resolution)
    FL_foot_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.3, 0.1]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    FR_foot_scanner = FL_foot_scanner.replace(prim_path="{ENV_REGEX_NS}/Robot/FR_foot")
    RL_foot_scanner = FL_foot_scanner.replace(prim_path="{ENV_REGEX_NS}/Robot/RL_foot")
    RR_foot_scanner = FL_foot_scanner.replace(prim_path="{ENV_REGEX_NS}/Robot/RR_foot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------


@configclass
class StairEventCfg:
    """Event specifications for the MDP."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.5, 1.5),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_from_stair,
        mode="reset",
        params={"lateral_range": (-0.3, 0.3), "yaw_range": (-math.pi / 4, math.pi / 4)},
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
    )


# -----------------------------------------------------------------------------
# Commands / Actions
# -----------------------------------------------------------------------------


@configclass
class StairCommandsCfg:
    """Command specifications for the MDP."""

    goal_command = mdp.StairPoseCommandCfg(goal_mode="stairs")


@configclass
class StairActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


# -----------------------------------------------------------------------------
# Observations (45 proprio + 4 goal + 32 foot heights + 441 height map = 522)
# -----------------------------------------------------------------------------


@configclass
class StairObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group (order preserved; the height map is last)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, clip=(-100, 100), noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        goal = ObsTerm(func=mdp.goal_pose_rel, clip=(-100, 100), params={"command_name": "goal_command"})
        foot_heights = ObsTerm(func=mdp.foot_heights_observation, clip=(-100, 100))
        height_map = ObsTerm(func=mdp.height_map_observation, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# Rewards (regularization shared by both stages; task rewards per stage)
# -----------------------------------------------------------------------------


@configclass
class StairRewardsCfg:
    """Regularization reward terms (arXiv 2602.03087v2, Table of reward weights)."""

    power = RewTerm(func=mdp.power_penalty, weight=-1e-4)
    stationary_power = RewTerm(func=mdp.stationary_power_penalty, weight=-1e-5)
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-1e-4)
    joint_limit = RewTerm(func=mdp.joint_position_limit_penalty, weight=-0.1)
    joint_vel_limit = RewTerm(func=mdp.joint_velocity_limit_penalty, weight=-0.1)
    joint_velocity = RewTerm(func=mdp.joint_velocity_penalty, weight=-5e-5)
    joint_acceleration = RewTerm(func=mdp.joint_acceleration_penalty, weight=-2e-8)
    thigh_collision = RewTerm(
        func=mdp.leg_contact_penalty,
        weight=-8.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )
    shank_foot_collision = RewTerm(
        func=mdp.leg_contact_penalty,
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_calf", ".*_foot"]), "threshold": 1.0},
    )
    flying = RewTerm(
        func=mdp.flying_penalty,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "air_time_threshold": 0.1},
    )
    step = RewTerm(
        func=mdp.step_reward,
        weight=0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 1.0},
    )


@configclass
class StairStage1RewardsCfg(StairRewardsCfg):
    """Stage 1: navigation rewards (coarse + fine) and a heading penalty."""

    nav_coarse = RewTerm(
        func=mdp.nav_progress_reward, weight=1.0, params={"command_name": "goal_command", "sigma": 5.0}
    )
    nav_fine = RewTerm(
        func=mdp.nav_progress_reward, weight=1.0, params={"command_name": "goal_command", "sigma": 1.0}
    )
    heading = RewTerm(func=mdp.heading_error_penalty, weight=-0.3, params={"command_name": "goal_command"})


@configclass
class StairStage2RewardsCfg(StairRewardsCfg):
    """Stage 2: centerline reward, path reward (modulated by the center term) and fine navigation."""

    center = RewTerm(func=mdp.centerline_reward, weight=0.1, params={"sigma": 0.5})
    path = RewTerm(func=mdp.path_progress_reward, weight=1.9, params={"sigma": 2.0, "max_progress": 1.5})
    nav_fine = RewTerm(
        func=mdp.nav_progress_reward, weight=1.0, params={"command_name": "goal_command", "sigma": 1.0}
    )


# -----------------------------------------------------------------------------
# Terminations / Curriculum
# -----------------------------------------------------------------------------


@configclass
class StairTerminationsCfg:
    """Termination terms for the MDP."""

    goal_reached = DoneTerm(func=mdp.goal_reached, params={"command_name": "goal_command"})
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class StairCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.stair_terrain_levels)


# -----------------------------------------------------------------------------
# Environment configurations
# -----------------------------------------------------------------------------


@configclass
class StairEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration shared by the two stair stages."""

    scene: StairSceneCfg = StairSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: StairObservationsCfg = StairObservationsCfg()
    actions: StairActionsCfg = StairActionsCfg()
    commands: StairCommandsCfg = StairCommandsCfg()
    rewards: StairRewardsCfg = StairRewardsCfg()
    terminations: StairTerminationsCfg = StairTerminationsCfg()
    events: StairEventCfg = StairEventCfg()
    curriculum: StairCurriculumCfg = StairCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # hold the same goal for the whole episode (resample only on reset)
        self.commands.goal_command.resampling_time_range = (self.episode_length_s, self.episode_length_s)

        # update sensor update periods (tick at the control rate)
        self.scene.contact_forces.update_period = self.sim.dt
        for scanner_name in ("height_scanner", "FL_foot_scanner", "FR_foot_scanner", "RL_foot_scanner", "RR_foot_scanner"):
            getattr(self.scene, scanner_name).update_period = self.decimation * self.sim.dt


@configclass
class StairStage1EnvCfg(StairEnvCfg):
    """Stage 1: pyramid stairs terrain with navigation rewards."""

    rewards: StairStage1RewardsCfg = StairStage1RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = stage1_terrain_cfg()
        self.commands.goal_command.goal_mode = "pyramid"
        # spawn at the platform center with random headings
        self.events.reset_base.params["lateral_range"] = (-0.2, 0.2)
        self.events.reset_base.params["yaw_range"] = (-math.pi, math.pi)


@configclass
class StairStage2EnvCfg(StairEnvCfg):
    """Stage 2: straight / L-shaped / spiral stair terrains with centerline rewards."""

    rewards: StairStage2RewardsCfg = StairStage2RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = stage2_terrain_cfg()


# -----------------------------------------------------------------------------
# Play configurations
# -----------------------------------------------------------------------------


@configclass
class StairStage1PlayEnvCfg(StairStage1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1


@configclass
class StairStage2PlayEnvCfg(StairStage2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 3
