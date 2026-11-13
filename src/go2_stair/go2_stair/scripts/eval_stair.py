# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained stair policy over 300 episodes per terrain configuration.

Implements the protocol of arXiv 2602.03087v2 (Sec. 6): for each step height in
4-14 cm (14 cm is out-of-distribution), 300 episodes are collected per terrain kind
and six metrics are reported: success rate, mean linear velocity, mean vertical
velocity, final position error, final heading error and mean power.

Metrics are accumulated per episode. The goal-reached termination is disabled so that
the final state at the moment of reaching the goal is observable; an episode counts as
successful when the goal pose is reached within the tolerances, at which point its
statistics are frozen (later falls do not pollute the metrics).

Usage:
    python eval_stair.py --task Unitree-Go2-Stair-Stage2 --checkpoint <model.pt> --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a stair policy (arXiv 2602.03087v2 protocol).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, required=True, help="Path to the model checkpoint.")
parser.add_argument("--num_episodes", type=int, default=300, help="Episodes per (height, kind) configuration.")
parser.add_argument("--num_envs", type=int, default=60, help="Number of parallel environments (multiple of the terrain kinds).")
parser.add_argument(
    "--step_heights", type=str, default="0.04,0.06,0.08,0.10,0.12,0.14",
    help="Comma-separated step heights in m (0.14 is out-of-distribution).",
)
parser.add_argument(
    "--terrains", type=str, default=None,
    help="Comma-separated terrain kinds to evaluate (default: all kinds of the task).",
)
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import os
from datetime import datetime

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import wrap_to_pi
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

# success tolerances (same as the goal_reached termination)
POS_TOL = 0.35
Z_TOL = 0.35
YAW_TOL = 0.45
# training step height range of the curriculum: [0.02, 0.12]
HEIGHT_MIN, HEIGHT_MAX = 0.02, 0.12


def make_eval_env(task: str, kind_names: list[str], step_height: float, num_envs: int, device: str):
    """Create an evaluation environment with a fixed terrain at the given step height."""
    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    # fixed terrain: single difficulty level mapped to the requested step height
    difficulty = (step_height - HEIGHT_MIN) / (HEIGHT_MAX - HEIGHT_MIN)
    gen_cfg = env_cfg.scene.terrain.terrain_generator
    gen_cfg.num_rows = 1
    gen_cfg.num_cols = len(kind_names)
    gen_cfg.difficulty_range = (difficulty, difficulty)
    gen_cfg.sub_terrains = {name: gen_cfg.sub_terrains[name] for name in kind_names}
    env_cfg.scene.terrain.max_init_terrain_level = 0
    # no curriculum moves and no goal-reached termination (success is detected by the eval loop)
    env_cfg.curriculum = None
    delattr(env_cfg.terminations, "goal_reached")
    # clean observations for evaluation
    env_cfg.observations.policy.enable_corruption = False
    if env_cfg.seed is None:
        env_cfg.seed = args_cli.seed
    return gym.make(task, cfg=env_cfg)


def load_policy(task: str, checkpoint: str, env, device: str):
    """Create an rsl-rl runner, load the checkpoint and return the inference policy."""
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint)
    return env, runner.get_inference_policy(device=env.unwrapped.device)


def main():
    """Run the evaluation protocol."""
    task = args_cli.task
    heights = [float(h) for h in args_cli.step_heights.split(",")]
    base_env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=None)
    all_kinds = list(base_env_cfg.scene.terrain.terrain_generator.sub_terrains.keys())
    if args_cli.terrains is not None:
        kinds = [k.strip() for k in args_cli.terrains.split(",")]
        for k in kinds:
            if k not in all_kinds:
                raise ValueError(f"Terrain kind '{k}' not in task terrains {all_kinds}.")
    else:
        kinds = all_kinds
    if args_cli.num_envs % len(kinds) != 0:
        raise ValueError(f"--num_envs must be a multiple of the number of terrain kinds ({len(kinds)}).")
    episodes_per_kind = args_cli.num_episodes // len(kinds)
    # print header
    print(f"Evaluating '{task}' terrains {kinds} at step heights {[f'{h:.2f}' for h in heights]}")
    print(f"Episodes per configuration: {args_cli.num_episodes} ({episodes_per_kind} per kind), seed={args_cli.seed}")

    # CSV output
    out_dir = os.path.join("logs", "rsl_rl", "eval_stair")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"eval_{task.lower().replace('-', '_')}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )
    csv_file = open(out_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["step_height", "terrain", "episodes", "success_rate",
         "mean_lin_vel", "mean_vert_vel", "mean_pos_err", "mean_heading_err", "mean_power"]
    )

    for height in heights:
        print(f"\n--- step height: {height:.2f} m ---")
        env, policy = None, None
        try:
            env = make_eval_env(task, kinds, height, args_cli.num_envs, args_cli.device)
            env, policy = load_policy(task, args_cli.checkpoint, env, args_cli.device)
            row = evaluate_configuration(env, policy, kinds, episodes_per_kind, csv_writer, height)
            print(
                f"[{height:.2f} m] success {row['success']:.1%} | lin_vel {row['lin_vel']:.3f} m/s | "
                f"vert_vel {row['vert_vel']:.3f} m/s | pos_err {row['pos_err']:.3f} m | "
                f"heading_err {row['heading_err']:.3f} rad | power {row['power']:.3f} W"
            )
        finally:
            if env is not None:
                env.close()

    csv_file.close()
    print(f"\nResults written to {out_path}")


def evaluate_configuration(env: RslRlVecEnvWrapper, policy, kinds: list[str], episodes_per_kind: int, csv_writer, height: float) -> dict:
    """Run episodes on one fixed terrain configuration and return aggregate metrics."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    cmd_term = unwrapped.command_manager.get_term("goal_command")
    dt = unwrapped.step_dt
    num_envs = unwrapped.num_envs
    device = unwrapped.device
    # per-env episode accounting
    acc_time = torch.zeros(num_envs, device=device)
    acc_lin = torch.zeros(num_envs, device=device)
    acc_vert = torch.zeros(num_envs, device=device)
    acc_power = torch.zeros(num_envs, device=device)
    reached_flag = torch.zeros(num_envs, dtype=torch.bool, device=device)
    finalized = torch.zeros(num_envs, dtype=torch.bool, device=device)
    last_pos_err = torch.zeros(num_envs, device=device)
    last_heading_err = torch.zeros(num_envs, device=device)

    # per-kind episode statistics
    kind_index = unwrapped.scene.terrain.terrain_types  # (num_envs,) -> sub-terrain column
    stats = {
        name: {"n_succ": 0, "n_episodes": 0, "lin": [], "vert": [], "pos": [], "heading": [], "power": []}
        for name in kinds
    }

    obs = env.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    pending_reset = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def finish_episodes(done_mask: torch.Tensor, success_mask: torch.Tensor):
        """Flush finalized episodes into the per-kind statistics."""
        nonlocal stats
        for i in done_mask.nonzero(as_tuple=False).flatten():
            name = kinds[int(kind_index[i])]
            s = stats[name]
            if s["n_episodes"] >= episodes_per_kind:
                continue
            s["n_episodes"] += 1
            if success_mask[i]:
                s["n_succ"] += 1
            s["lin"].append(float(acc_lin[i] / acc_time[i].clamp(min=dt)))
            s["vert"].append(float(acc_vert[i] / acc_time[i].clamp(min=dt)))
            s["pos"].append(float(last_pos_err[i]))
            s["heading"].append(float(last_heading_err[i]))
            s["power"].append(float(acc_power[i] / acc_time[i].clamp(min=dt)))

    with torch.inference_mode():
        while any(s["n_episodes"] < episodes_per_kind for s in stats.values()):
            # ---- observe the current state (start of episode for envs reset last step) ----
            root_pos = robot.data.root_pos_w
            d_xy = torch.norm(cmd_term.pos_command_w[:, :2] - root_pos[:, :2], dim=1)
            d_z = torch.abs(cmd_term.pos_command_w[:, 2] - root_pos[:, 2])
            yaw_err = torch.abs(wrap_to_pi(cmd_term.heading_command_w - robot.data.heading_w))
            lin_speed = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
            vert_speed = torch.abs(robot.data.root_lin_vel_b[:, 2])
            power = torch.abs(robot.data.applied_torque * robot.data.joint_vel).sum(dim=1)
            # reset accumulators for episodes that started this step
            acc_time[pending_reset] = 0.0
            acc_lin[pending_reset] = 0.0
            acc_vert[pending_reset] = 0.0
            acc_power[pending_reset] = 0.0
            reached_flag[pending_reset] = False
            finalized[pending_reset] = False
            # accumulate for active episodes
            active = ~finalized & ~pending_reset
            acc_time[active] += dt
            acc_lin[active] += lin_speed[active] * dt
            acc_vert[active] += vert_speed[active] * dt
            acc_power[active] += power[active] * dt
            last_pos_err[active] = torch.norm(
                cmd_term.pos_command_w[active] - root_pos[active], dim=1
            )
            last_heading_err[active] = yaw_err[active]
            # success: goal pose reached within tolerances -> freeze the episode
            reached = active & (d_xy < POS_TOL) & (d_z < Z_TOL) & (yaw_err < YAW_TOL)
            if reached.any():
                finish_episodes(reached, torch.ones_like(reached))
                finalized[reached] = True
                reached_flag[reached] = True
            # ---- step the environment ----
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            obs = obs[0] if isinstance(obs, tuple) else obs
            pending_reset = dones
            # failure/time-out: flush episodes that ended without reaching the goal
            failed = dones & ~finalized
            if failed.any():
                finish_episodes(failed, torch.zeros_like(failed))
                finalized[failed] = True

    # aggregate and write CSV
    totals = {"lin": [], "vert": [], "pos": [], "heading": [], "power": []}
    n_succ_total, n_total = 0, 0
    for name in kinds:
        s = stats[name]
        n_succ_total += s["n_succ"]
        n_total += s["n_episodes"]
        for key in totals:
            totals[key] += s[key]
        csv_writer.writerow([
            f"{height:.2f}", name, s["n_episodes"],
            f"{s['n_succ'] / max(s['n_episodes'], 1):.4f}",
            f"{sum(s['lin']) / max(s['n_episodes'], 1):.4f}",
            f"{sum(s['vert']) / max(s['n_episodes'], 1):.4f}",
            f"{sum(s['pos']) / max(s['n_episodes'], 1):.4f}",
            f"{sum(s['heading']) / max(s['n_episodes'], 1):.4f}",
            f"{sum(s['power']) / max(s['n_episodes'], 1):.4f}",
        ])
    return {
        "success": n_succ_total / max(n_total, 1),
        "lin_vel": sum(totals["lin"]) / max(n_total, 1),
        "vert_vel": sum(totals["vert"]) / max(n_total, 1),
        "pos_err": sum(totals["pos"]) / max(n_total, 1),
        "heading_err": sum(totals["heading"]) / max(n_total, 1),
        "power": sum(totals["power"]) / max(n_total, 1),
    }


if __name__ == "__main__":
    main()
    simulation_app.close()
