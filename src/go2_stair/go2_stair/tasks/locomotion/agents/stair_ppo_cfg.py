# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class StairPPORunnerCfg(BasePPORunnerCfg):
    """PPO runner configuration for the stair tasks (height map CNN policy).

    Both stages share the same network structure so that stage 2 can resume from the
    stage 1 checkpoint.
    """

    policy = RslRlPpoActorCriticCfg(
        class_name="HeightmapActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )


@configclass
class StairStage1PPORunnerCfg(StairPPORunnerCfg):
    experiment_name = "go2_stair_stage1"


@configclass
class StairStage2PPORunnerCfg(StairPPORunnerCfg):
    experiment_name = "go2_stair_stage2"
