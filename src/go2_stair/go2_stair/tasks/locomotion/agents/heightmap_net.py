"""Actor-critic with a shared CNN height map encoder (arXiv 2602.03087v2).

The policy consumes 522-dim observations: 81 proprioceptive values (velocities, gravity,
joints, actions, goal) + 32 foot heights + a 21x21 (441) height map. The last 441 values
are reshaped into the height map and encoded by a shared CNN
(3 conv layers 8 -> 16 -> 32, 2x2 max pooling, adaptive pooling to a 128-dim vector),
then concatenated with the proprioceptive part and passed through MLP heads.

rsl-rl resolves the policy class with ``eval`` inside the ``rsl_rl.runners.on_policy_runner``
module namespace, so the class is registered there at import time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic
from rsl_rl.utils import resolve_nn_activation

HEIGHTMAP_ROWS = 21
HEIGHTMAP_COLS = 21
HEIGHTMAP_SIZE = HEIGHTMAP_ROWS * HEIGHTMAP_COLS  # 441
PROPRIO_DIM = 81  # 45 proprio + 4 goal + 32 foot heights
CNN_OUT_DIM = 128


class _HeightmapHead(nn.Module):
    """MLP head that combines the proprioceptive part with the shared CNN encoding."""

    def __init__(
        self,
        proprio_dim: int,
        cnn: nn.Module,
        cnn_out_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        activation: nn.Module,
    ):
        super().__init__()
        self.cnn = cnn
        self.proprio_dim = proprio_dim
        layers: list[nn.Module] = [nn.Linear(proprio_dim + cnn_out_dim, hidden_dims[0]), activation]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), activation]
        layers += [nn.Linear(hidden_dims[-1], out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        heightmap = observations[:, -HEIGHTMAP_SIZE:].reshape(-1, 1, HEIGHTMAP_ROWS, HEIGHTMAP_COLS)
        encoded = self.cnn(heightmap)
        x = torch.cat([observations[:, : self.proprio_dim], encoded], dim=-1)
        return self.mlp(x)


class HeightmapActorCritic(ActorCritic):
    """Actor-critic whose actor and critic share a CNN height map encoder.

    The rsl-rl 3.x constructor receives the observation dict and group mapping; both
    heads consume the full "policy" group (proprio + height map concatenated).
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [128, 128, 64],
        critic_hidden_dims: list[int] = [128, 128, 64],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        act = resolve_nn_activation(activation)
        # shared CNN encoder: 3 conv layers 8 -> 16 -> 32 with 2x2 max pooling,
        # adaptive average pooling to a 2x2 grid -> 128-dim encoding
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            act,
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            act,
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            act,
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.actor = _HeightmapHead(PROPRIO_DIM, self.cnn_encoder, CNN_OUT_DIM, actor_hidden_dims, num_actions, act)
        self.critic = _HeightmapHead(PROPRIO_DIM, self.cnn_encoder, CNN_OUT_DIM, critic_hidden_dims, 1, act)

        print(f"Actor (heightmap): {self.actor}")
        print(f"Critic (heightmap): {self.critic}")


# rsl-rl OnPolicyRunner resolves the policy class via eval() in the module globals of
# rsl_rl.runners.on_policy_runner; register the class there so class_name="HeightmapActorCritic" works.
import rsl_rl.runners.on_policy_runner as _rsl_rl_on_policy_runner  # noqa: E402

_rsl_rl_on_policy_runner.HeightmapActorCritic = HeightmapActorCritic
