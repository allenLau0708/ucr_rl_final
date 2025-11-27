"""
Configuration for Fuel Collection RL Project.
CS258 Final Project
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PPOConfig:
    """PPO algorithm configuration."""
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    batch_size: int = 64
    n_epochs: int = 10
    n_steps: int = 2048
    total_timesteps: int = 100000
    device: str = "cpu"
