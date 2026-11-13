"""楼梯任务注册桥：覆盖 unitree_rl_lab 的 go2 注册文件。

部署时拷贝到：
  stack/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/__init__.py

train.py / list_envs.py 只遍历 unitree_rl_lab 的 robots 包来触发 gym.register，
楼梯任务的实现位于本项目 go2_stair 包（src/go2_stair），这里只做注册桥。
"""

import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Stair-Stage1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_stair.tasks.locomotion.robots.go2.stair_env_cfg:StairStage1EnvCfg",
        "play_env_cfg_entry_point": "go2_stair.tasks.locomotion.robots.go2.stair_env_cfg:StairStage1PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "go2_stair.tasks.locomotion.agents.stair_ppo_cfg:StairStage1PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Stair-Stage2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_stair.tasks.locomotion.robots.go2.stair_env_cfg:StairStage2EnvCfg",
        "play_env_cfg_entry_point": "go2_stair.tasks.locomotion.robots.go2.stair_env_cfg:StairStage2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "go2_stair.tasks.locomotion.agents.stair_ppo_cfg:StairStage2PPORunnerCfg",
    },
)
