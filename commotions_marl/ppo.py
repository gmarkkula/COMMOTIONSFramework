"""TorchRL TensorDictModule PPO policy/critic trainer for CrossingGlanceGymEnv."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from .env import CrossingGlanceGymEnv


class _ActorBackbone(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.loc_head = nn.Linear(hidden_dim, act_dim)
        self.log_scale_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(observation)
        loc = self.loc_head(h)
        scale = torch.nn.functional.softplus(self.log_scale_head(h)) + 1e-3
        return loc, scale


def build_ppo_actor_critic(
    obs_dim: int = 8,
    act_dim: int = 2,
    hidden_dim: int = 64,
    device: str | torch.device = "cpu",
) -> Tuple[ProbabilisticActor, ValueOperator]:
    """Construct TorchRL ``TensorDictModule`` actor and critic operators."""
    dev = torch.device(device)
    policy_Param_module = TensorDictModule(
        _ActorBackbone(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim).to(dev),
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=policy_Param_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": torch.full((act_dim,), -1.0, device=dev),
            "high": torch.full((act_dim,), 1.0, device=dev),
        },
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    )

    critic_mlp = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    ).to(dev)
    critic = ValueOperator(
        module=critic_mlp,
        in_keys=["observation"],
        out_keys=["state_value"],
    )
    return actor, critic


class TorchRLCrossingPPOTrainer:
    """Lightweight TorchRL ``ClipPPOLoss`` + ``GAE`` trainer for road crossing."""

    def __init__(
        self,
        env: CrossingGlanceGymEnv | None = None,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        clip_epsilon: float = 0.2,
        device: str = "cpu",
    ) -> None:
        self.env = env if env is not None else CrossingGlanceGymEnv()
        self.device = torch.device(device)
        obs_dim = int(self.env.observation_space.shape[0])
        act_dim = int(self.env.action_space.shape[0])

        self.actor, self.critic = build_ppo_actor_critic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            device=self.device,
        )
        self.advantage_module = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=self.critic,
            average_gae=True,
        )
        self.loss_module = ClipPPOLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            clip_epsilon=clip_epsilon,
            entropy_bonus=True,
            entropy_coeff=0.01,
        )
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), lr=lr)

    def collect_rollout(self, n_steps: int = 64, seed: int | None = 0) -> TensorDict:
        obs_list, act_list, logp_list, rew_list, done_list, next_obs_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        obs_np, _ = self.env.reset(seed=seed)
        for _ in range(n_steps):
            obs_tensor = torch.as_tensor(
                obs_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            td_in = TensorDict(
                {"observation": obs_tensor},
                batch_size=[1],
                device=self.device,
            )
            with torch.no_grad():
                td_out = self.actor(td_in)
            act_t = td_out["action"].squeeze(0)
            logp_t = td_out["sample_log_prob"].squeeze(0)

            next_obs_np, rew, terminated, truncated, _ = self.env.step(
                act_t.detach().cpu().numpy()
            )
            done = bool(terminated or truncated)

            obs_list.append(torch.as_tensor(obs_np, dtype=torch.float32, device=self.device))
            act_list.append(act_t)
            logp_list.append(logp_t)
            rew_list.append(torch.tensor([rew], dtype=torch.float32, device=self.device))
            done_list.append(torch.tensor([done], dtype=torch.bool, device=self.device))
            next_obs_list.append(
                torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.device)
            )

            if done:
                obs_np, _ = self.env.reset()
            else:
                obs_np = next_obs_np

        batch = TensorDict(
            {
                "observation": torch.stack(obs_list, dim=0),
                "action": torch.stack(act_list, dim=0),
                "sample_log_prob": torch.stack(logp_list, dim=0),
                "next": TensorDict(
                    {
                        "observation": torch.stack(next_obs_list, dim=0),
                        "reward": torch.stack(rew_list, dim=0),
                        "done": torch.stack(done_list, dim=0),
                        "terminated": torch.stack(done_list, dim=0),
                    },
                    batch_size=[n_steps],
                    device=self.device,
                ),
            },
            batch_size=[n_steps],
            device=self.device,
        )
        return batch

    def train_step(
        self, rollout_steps: int = 64, ppo_epochs: int = 2, seed: int | None = 0
    ) -> Dict[str, float]:
        rollout = self.collect_rollout(n_steps=rollout_steps, seed=seed)
        with torch.no_grad():
            self.advantage_module(rollout)

        total_loss_val = 0.0
        for _ in range(ppo_epochs):
            loss_td = self.loss_module(rollout)
            loss = (
                loss_td["loss_objective"]
                + loss_td["loss_critic"]
                + loss_td.get("loss_entropy", 0.0)
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), 1.0)
            self.optimizer.step()
            total_loss_val = float(loss.detach().cpu().item())

        mean_reward = float(rollout["next", "reward"].mean().detach().cpu().item())
        return {
            "total_loss": total_loss_val,
            "mean_step_reward": mean_reward,
            "rollout_steps": float(rollout_steps),
        }
