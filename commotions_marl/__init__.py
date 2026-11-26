"""Optional bounded-rationality perceptual-motor MARL subpackage for COMMOTIONS.

Implements Wang, Dogar & Markkula (2025, arXiv:2510.27383) single-agent
Gymnasium and multi-agent PettingZoo road-crossing environments together with
lazy-loaded TorchRL ``TensorDictModule`` PPO training utilities so core
``commotions.py`` remains strictly zero-PyTorch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .env import CrossingGlanceGymEnv, PerceptualMotorParallelEnv
    from .ppo import TorchRLCrossingPPOTrainer, build_ppo_actor_critic

__all__ = [
    "CrossingGlanceGymEnv",
    "PerceptualMotorParallelEnv",
    "TorchRLCrossingPPOTrainer",
    "build_ppo_actor_critic",
]


def __getattr__(name: str) -> Any:
    if name in ("CrossingGlanceGymEnv", "PerceptualMotorParallelEnv"):
        from . import env as _env

        return getattr(_env, name)
    if name in ("TorchRLCrossingPPOTrainer", "build_ppo_actor_critic"):
        from . import ppo as _ppo

        return getattr(_ppo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
