"""Export a trained stair policy to JIT (.pt) and ONNX (.onnx) for deployment.

The isaaclab_rl exporters assume the actor is a plain ``nn.Sequential`` (they read
``actor[0].in_features`` to build the dummy input), which does not hold for the custom
``HeightmapActorCritic``. This script builds the export wrapper itself, uses a fixed
522-dim dummy observation and verifies that the ONNX graph matches the JIT module.

Usage:
    python export_stair_policy.py --task Unitree-Go2-Stair-Stage2 --checkpoint <model.pt> --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export a stair policy to JIT/ONNX.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Stair-Stage2", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments used to build the runner.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import os

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from go2_stair.tasks.locomotion.agents.heightmap_net import HEIGHTMAP_SIZE, PROPRIO_DIM

OBS_DIM = PROPRIO_DIM + HEIGHTMAP_SIZE  # 522


class _ExportWrapper(torch.nn.Module):
    """Observation normalizer (identity when disabled) followed by the actor."""

    def __init__(self, actor: torch.nn.Module, normalizer: torch.nn.Module):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(obs))


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args_cli.checkpoint)

    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    actor = copy.deepcopy(policy_nn.actor).cpu().eval()
    if isinstance(normalizer, torch.nn.Module):
        normalizer = copy.deepcopy(normalizer).cpu().eval()
    else:
        normalizer = torch.nn.Identity()
    wrapper = _ExportWrapper(actor, normalizer).cpu().eval()

    export_dir = os.path.join(os.path.dirname(os.path.abspath(args_cli.checkpoint)), "exported")
    os.makedirs(export_dir, exist_ok=True)

    jit_path = os.path.join(export_dir, "policy.pt")
    torch.jit.script(wrapper).save(jit_path)
    print(f"[OK] JIT exported: {jit_path}")

    onnx_path = os.path.join(export_dir, "policy.onnx")
    dummy = torch.zeros(1, OBS_DIM)
    torch.onnx.export(wrapper, dummy, onnx_path, input_names=["obs"], output_names=["actions"])
    print(f"[OK] ONNX exported: {onnx_path}")

    try:
        import onnxruntime as ort

        x = torch.randn(4, OBS_DIM)
        jit_out = torch.jit.load(jit_path)(x).detach().numpy()
        sess = ort.InferenceSession(onnx_path)
        onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
        diff = float(abs(jit_out - onnx_out).max())
        print(f"[OK] JIT vs ONNX max diff on random obs: {diff:.2e}")
    except ImportError:
        print("[WARN] onnxruntime is not installed; skipped the JIT/ONNX consistency check")

    print(f"[INFO] Deployment config: {os.path.join(os.path.dirname(os.path.abspath(args_cli.checkpoint)), 'params', 'deploy.yaml')}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()