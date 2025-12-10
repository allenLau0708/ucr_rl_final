"""
Multi-Critic PPO Agent for Fuel Collection Environment.
CS258 Final Project: Learning to Survive in a Dual-Objective Environment

Multi-Objective RL: Separate critic per reward term to avoid exploiting easy terms.
Now with an explicit SHARED BACKBONE: one feature encoder shared by actor and all critics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List
from collections import deque
import gymnasium as gym

from config import PPOConfig


# =========================
#  Networks
# =========================

class SharedBackbone(nn.Module):
    """Shared feature extractor used by both actor and all critics."""

    def __init__(self, obs_dim: int, hidden_sizes: List[int] = [256, 256]):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MultiCriticNetwork(nn.Module):
    """
    Actor + multiple critics with an explicit SHARED BACKBONE:

        obs -> backbone -> shared feature
             -> actor_head -> action logits
             -> value_heads[term] -> V_term(s)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int],
        reward_terms: List[str],
    ):
        super().__init__()

        # Shared feature encoder
        self.backbone = SharedBackbone(obs_dim, hidden_sizes)

        # Actor head: shared feature -> action logits
        last_dim = hidden_sizes[-1]
        self.actor_head = nn.Linear(last_dim, action_dim)

        # Critic heads: shared feature -> scalar value for each reward term
        self.value_heads = nn.ModuleDict({
            term: nn.Linear(last_dim, 1) for term in reward_terms
        })

        self.reward_terms = reward_terms
        self._init_heads()

    def _init_heads(self):
        # Initialize actor + critic heads
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)
        for head in self.value_heads.values():
            nn.init.orthogonal_(head.weight, gain=1.0)
            nn.init.constant_(head.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """
        Returns:
            logits: (B, action_dim)
            values_dict: {term: (B,)} for each reward term
        """
        feat = self.backbone(obs)
        logits = self.actor_head(feat)
        values = {
            term: head(feat).squeeze(-1)
            for term, head in self.value_heads.items()
        }
        return logits, values

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            feat = self.backbone(obs)
            logits = self.actor_head(feat)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()

            # For logging: we also compute critic values (but only mean scalar)
            values = {
                term: head(feat).squeeze(-1)
                for term, head in self.value_heads.items()
            }
            mean_value = torch.stack(list(values.values())).mean().item()

            return action.item(), dist.log_prob(action).item(), mean_value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Used in PPO update.

        Returns:
            log_probs: (B,)
            values_dict: {term: (B,)}
            entropy: (B,)
        """
        feat = self.backbone(obs)
        logits = self.actor_head(feat)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        values = {
            term: head(feat).squeeze(-1)
            for term, head in self.value_heads.items()
        }
        return log_probs, values, entropy


# =========================
#  Rollout Buffer
# =========================

class MultiRewardRolloutBuffer:
    """Buffer for multi-critic with decomposed rewards."""

    def __init__(self, reward_terms: List[str]):
        self.reward_terms = reward_terms
        self.reset()

    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = {term: [] for term in self.reward_terms}
        self.values = {term: [] for term in self.reward_terms}
        self.log_probs = []
        self.dones = []

    def add(
        self,
        obs,
        action,
        reward_dict: Dict[str, float],
        value_dict: Dict[str, float],
        log_prob,
        done,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        for term in self.reward_terms:
            self.rewards[term].append(reward_dict.get(term, 0.0))
            self.values[term].append(value_dict.get(term, 0.0))
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def __len__(self):
        return len(self.observations)


# =========================
#  Multi-Critic PPO Agent
# =========================

class MultiCriticPPOAgent:
    """Multi-Critic PPO agent: separate critic per reward term with shared backbone."""

    def __init__(self, env: gym.Env, config: PPOConfig, reward_scheme=None):
        self.env = env
        self.config = config

        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device(config.device)

        # Reward terms: goal, fuel, survival (step + death + collision + revisit)
        self.reward_terms = ['goal', 'fuel', 'survival']

        # Network with shared backbone
        self.network = MultiCriticNetwork(
            self.obs_dim,
            self.action_dim,
            config.hidden_sizes,
            self.reward_terms,
        ).to(self.device)

        # ---- Optimizers ----
        # Actor optimizer updates backbone + actor head
        actor_params = list(self.network.backbone.parameters()) + \
                       list(self.network.actor_head.parameters())
        self.actor_optimizer = optim.Adam(
            actor_params,
            lr=config.learning_rate,
        )

        # Each critic optimizer only updates its own value head
        self.critic_optimizers = {
            term: optim.Adam(
                self.network.value_heads[term].parameters(),
                lr=config.learning_rate * 2,  # Critic LR can be higher
            )
            for term in self.reward_terms
        }

        self.buffer = MultiRewardRolloutBuffer(self.reward_terms)

        self.total_steps = 0
        self.episode_count = 0
        self.stats = {
            'episode_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'fuel_collected': deque(maxlen=100),
            'reached_goal': deque(maxlen=100),
            'died': deque(maxlen=100),
        }

    # ----------------------
    # Reward decomposition
    # ----------------------
    def _decompose_reward(
        self,
        reward: float,
        info: dict,
        prev_fuel_collected: int,
        current_fuel_collected: int,
    ) -> Dict[str, float]:
        """Decompose scalar reward into reward terms based on environment config."""
        reward_dict = {term: 0.0 for term in self.reward_terms}

        # Env reward constants
        goal_reward = self.env.config.goal_reward
        fuel_reward = self.env.config.fuel_pickup_reward
        death_penalty = self.env.config.death_penalty

        # Goal reward (only when reaching goal)
        if info.get('reached_goal', False):
            reward_dict['goal'] = goal_reward

        # Fuel reward (when fuel is collected)
        if current_fuel_collected > prev_fuel_collected:
            reward_dict['fuel'] = fuel_reward

        # Survival reward: everything else (step_penalty, death, collision, revisit)
        if info.get('died', False):
            reward_dict['survival'] = death_penalty
        else:
            # Approximate: total reward minus explicit goal and fuel parts
            reward_dict['survival'] = reward - reward_dict['goal'] - reward_dict['fuel']

        return reward_dict

    # ----------------------
    # Rollout collection
    # ----------------------
    def collect_rollouts(self, n_steps: int) -> Dict[str, float]:
        self.buffer.reset()
        obs, info = self.env.reset()

        ep_reward = 0.0
        ep_length = 0
        prev_fuel_collected = 0

        for _ in range(n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, _ = self.network.get_action(obs_tensor)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Current fuel collected
            current_fuel_collected = info.get('fuel_collected', 0)

            # Decompose reward based on environment state
            reward_dict = self._decompose_reward(
                reward,
                info,
                prev_fuel_collected,
                current_fuel_collected,
            )

            # Get values from all critics (scalar)
            with torch.no_grad():
                _, value_dict = self.network.forward(obs_tensor)
                value_dict_scalar = {k: v.item() for k, v in value_dict.items()}

            # Store transition
            self.buffer.add(
                obs,
                action,
                reward_dict,
                value_dict_scalar,
                log_prob,
                done,
            )

            ep_reward += reward
            ep_length += 1
            self.total_steps += 1

            if done:
                self.stats['episode_rewards'].append(ep_reward)
                self.stats['episode_lengths'].append(ep_length)
                self.stats['fuel_collected'].append(current_fuel_collected)
                self.stats['reached_goal'].append(
                    1.0 if info.get('reached_goal') else 0.0
                )
                self.stats['died'].append(1.0 if info.get('died') else 0.0)

                self.episode_count += 1
                obs, info = self.env.reset()
                ep_reward = 0.0
                ep_length = 0
                prev_fuel_collected = 0
            else:
                obs = next_obs
                prev_fuel_collected = current_fuel_collected

        self._compute_multi_gae()

        return {
            'mean_reward': np.mean(self.stats['episode_rewards'])
            if self.stats['episode_rewards']
            else 0.0,
            'mean_length': np.mean(self.stats['episode_lengths'])
            if self.stats['episode_lengths']
            else 0.0,
            'mean_fuel': np.mean(self.stats['fuel_collected'])
            if self.stats['fuel_collected']
            else 0.0,
            'goal_rate': np.mean(self.stats['reached_goal'])
            if self.stats['reached_goal']
            else 0.0,
            'death_rate': np.mean(self.stats['died'])
            if self.stats['died']
            else 0.0,
        }

    # ----------------------
    # GAE per reward term
    # ----------------------
    def _compute_multi_gae(self):
        """Compute GAE for each reward term separately."""
        gamma, lam = self.config.gamma, self.config.gae_lambda

        self.returns = {}
        self.advantages = {}

        for term in self.reward_terms:
            rewards = np.array(self.buffer.rewards[term], dtype=np.float32)
            values = np.array(self.buffer.values[term], dtype=np.float32)
            dones = np.array(self.buffer.dones, dtype=np.float32)

            n = len(rewards)
            advantages = np.zeros(n, dtype=np.float32)
            last_gae = 0.0

            # Last value (bootstrap if not done)
            if dones[-1]:
                last_value = 0.0
            else:
                last_obs = torch.FloatTensor(self.buffer.observations[-1]).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.network.backbone(last_obs)
                    last_value = self.network.value_heads[term](feat).item()

            for t in reversed(range(n)):
                if t == n - 1:
                    next_value = last_value
                    next_done = dones[t]
                else:
                    next_value = values[t + 1]
                    next_done = dones[t + 1]

                delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
                last_gae = delta + gamma * lam * (1 - next_done) * last_gae
                advantages[t] = last_gae

            self.returns[term] = advantages + values

            # Normalize advantages per term
            mean = advantages.mean()
            std = advantages.std()
            norm_adv = (advantages - mean) / (std + 1e-8)
            self.advantages[term] = norm_adv

    # ----------------------
    # PPO update
    # ----------------------
    def update(self) -> Dict[str, float]:
        obs = torch.FloatTensor(np.array(self.buffer.observations)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)

        returns = {
            term: torch.FloatTensor(self.returns[term]).to(self.device)
            for term in self.reward_terms
        }
        advantages = {
            term: torch.FloatTensor(self.advantages[term]).to(self.device)
            for term in self.reward_terms
        }

        n = len(obs)
        total_loss = 0.0
        n_updates = 0

        for _ in range(self.config.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.config.batch_size):
                idx = indices[start:start + self.config.batch_size]

                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_old_logp = old_log_probs[idx]

                new_log_probs, values_dict, entropy = self.network.evaluate_actions(
                    batch_obs, batch_actions
                )

                # Combine normalized advantages from all critics (equal weights)
                combined_adv = torch.zeros_like(
                    advantages[self.reward_terms[0]][idx]
                )
                for term in self.reward_terms:
                    combined_adv += advantages[term][idx] / len(self.reward_terms)

                ratio = torch.exp(new_log_probs - batch_old_logp)
                surr1 = ratio * combined_adv
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * combined_adv

                policy_loss = -torch.min(surr1, surr2).mean()
                actor_loss = policy_loss - self.config.entropy_coef * entropy.mean()

                # ---- Update actor (backbone + actor_head) ----
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(
                    list(self.network.backbone.parameters()) +
                    list(self.network.actor_head.parameters()),
                    self.config.max_grad_norm,
                )
                self.actor_optimizer.step()

                # ---- Update critics (value heads) ----
                total_critic_loss = 0.0
                for term in self.reward_terms:
                    v_pred = values_dict[term]
                    v_target = returns[term][idx]
                    critic_loss = nn.functional.mse_loss(v_pred, v_target)

                    self.critic_optimizers[term].zero_grad()
                    critic_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(
                        self.network.value_heads[term].parameters(),
                        self.config.max_grad_norm,
                    )
                    self.critic_optimizers[term].step()

                    total_critic_loss += critic_loss.item()

                total_loss += actor_loss.item() + total_critic_loss
                n_updates += 1

        return {'loss': total_loss / max(1, n_updates)}

    # ----------------------
    # Evaluate / Predict
    # ----------------------
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, _, _ = self.network.get_action(obs_tensor, deterministic)
        return action

    def evaluate(self, n_episodes: int = 100) -> Dict[str, float]:
        """Deterministic evaluation of the learned policy."""
        results = {'goal': 0, 'death': 0, 'fuel': [], 'steps': [], 'reward': []}

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

            if info.get('reached_goal'):
                results['goal'] += 1
            if info.get('died'):
                results['death'] += 1

            results['fuel'].append(info.get('fuel_collected', 0))
            results['steps'].append(info.get('steps_taken', 0))
            results['reward'].append(total_reward)

        return {
            'goal_rate': results['goal'] / n_episodes,
            'death_rate': results['death'] / n_episodes,
            'mean_fuel': np.mean(results['fuel']),
            'mean_steps': np.mean(results['steps']),
            'mean_reward': np.mean(results['reward']),
        }

    # ----------------------
    # Save / Load
    # ----------------------
    def save(self, path: str):
        torch.save(
            {
                'network': self.network.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizers': {
                    term: opt.state_dict()
                    for term, opt in self.critic_optimizers.items()
                },
                'config': self.config,
                'reward_terms': self.reward_terms,
            },
            path,
        )
        print(f"Saved: {path}")

    def load(self, path: str):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(data['network'])
        if 'actor_optimizer' in data:
            self.actor_optimizer.load_state_dict(data['actor_optimizer'])
            for term in self.reward_terms:
                if term in data['critic_optimizers']:
                    self.critic_optimizers[term].load_state_dict(
                        data['critic_optimizers'][term]
                    )
        print(f"Loaded: {path}")
