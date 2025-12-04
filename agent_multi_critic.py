"""
Multi-Critic PPO Agent for Fuel Collection Environment.
CS258 Final Project: Learning to Survive in a Dual-Objective Environment

Multi-Objective RL: Separate critic per reward term to avoid exploiting easy terms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional
from collections import deque
import gymnasium as gym

from config import PPOConfig


class ActorNetwork(nn.Module):
    """Actor network - outputs action probabilities."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: List[int] = [256, 256]):
        super().__init__()
        
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        layers.append(nn.Linear(hidden_sizes[-1], action_dim))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # Last layer uses smaller init for stable training
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)  # Returns logits


class CriticNetwork(nn.Module):
    """Critic network - outputs state value for a specific reward term."""
    
    def __init__(self, obs_dim: int, hidden_sizes: List[int] = [256, 256]):
        super().__init__()
        
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.network[-1].weight, gain=1.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs).squeeze(-1)  # Returns value


class MultiCriticNetwork(nn.Module):
    """Multi-Critic: Separate critic for each reward term."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: List[int] = [256, 256], 
                 reward_terms: List[str] = None):
        super().__init__()
        
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_sizes)
        
        # Default reward terms: goal, fuel, survival
        if reward_terms is None:
            reward_terms = ['goal', 'fuel', 'survival']
        
        self.reward_terms = reward_terms
        self.critics = nn.ModuleDict({
            term: CriticNetwork(obs_dim, hidden_sizes) 
            for term in reward_terms
        })
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = self.actor(obs)
        values = {term: critic(obs) for term, critic in self.critics.items()}
        return logits, values
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            logits = self.actor(obs)
            values = {term: critic(obs) for term, critic in self.critics.items()}
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = torch.argmax(probs, dim=-1) if deterministic else dist.sample()
            # Return mean value for compatibility
            mean_value = torch.stack(list(values.values())).mean().item()
            return action.item(), dist.log_prob(action).item(), mean_value
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits = self.actor(obs)
        values = {term: critic(obs) for term, critic in self.critics.items()}
        dist = Categorical(torch.softmax(logits, dim=-1))
        return dist.log_prob(actions), values, dist.entropy()


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
    
    def add(self, obs, action, reward_dict: Dict[str, float], value_dict: Dict[str, float], 
            log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        for term in self.reward_terms:
            self.rewards[term].append(reward_dict.get(term, 0.0))
            self.values[term].append(value_dict.get(term, 0.0))
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def __len__(self):
        return len(self.observations)


class MultiCriticPPOAgent:
    """Multi-Critic PPO agent: separate critic per reward term."""
    
    def __init__(self, env: gym.Env, config: PPOConfig, reward_scheme=None):
        self.env = env
        self.config = config
        
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device(config.device)
        
        # Reward terms: goal, fuel, survival (step + death + collision + revisit)
        self.reward_terms = ['goal', 'fuel', 'survival']
        
        # Advantage weights for combining different objectives
        # Following the paper: simple equal weights (1/K) often work well
        # But we can adjust based on task importance
        # Paper suggests: ω_k = 1/K for K objectives, but allows customization
        self.advantage_weights = {
            'goal': 1.0,      # Most important: reaching goal
            'fuel': 1.0,      # Important: collecting fuel (equal weight as paper suggests)
            'survival': 1.0   # Base survival (equal weight - let normalization handle scale)
        }
        # Normalize weights to sum to 1 (as per paper: Σ_k ω_k = 1)
        total_weight = sum(self.advantage_weights.values())
        self.advantage_weights = {k: v / total_weight for k, v in self.advantage_weights.items()}
        
        self.network = MultiCriticNetwork(
            self.obs_dim, self.action_dim, config.hidden_sizes, self.reward_terms
        ).to(self.device)
        
        # Separate optimizer for actor
        self.actor_optimizer = optim.Adam(
            self.network.actor.parameters(), 
            lr=config.learning_rate
        )
        
        # Separate optimizer for each critic with unified learning rates
        # Following the paper: use same learning rate for all critics for stability
        # The normalization handles scale differences, so we don't need different LRs
        critic_lr = config.learning_rate * 1.5  # Slightly higher than actor, but not too high
        self.critic_optimizers = {
            'goal': optim.Adam(
                self.network.critics['goal'].parameters(),
                lr=critic_lr
            ),
            'fuel': optim.Adam(
                self.network.critics['fuel'].parameters(),
                lr=critic_lr
            ),
            'survival': optim.Adam(
                self.network.critics['survival'].parameters(),
                lr=critic_lr
            )
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
        
        # Track reward decomposition for monitoring
        self.reward_decomposition_stats = {
            term: deque(maxlen=1000) for term in self.reward_terms
        }
    
    def _decompose_reward(self, reward: float, info: dict, prev_fuel_collected: int, 
                         current_fuel_collected: int) -> Dict[str, float]:
        """
        Decompose scalar reward into reward terms based on environment config.
        Improved version: more accurate decomposition.
        """
        reward_dict = {term: 0.0 for term in self.reward_terms}
        
        # Get config values
        goal_reward = self.env.config.goal_reward
        fuel_reward = self.env.config.fuel_pickup_reward
        death_penalty = self.env.config.death_penalty
        step_penalty = self.env.config.step_penalty
        
        # Goal reward (only when reaching goal) - sparse, high value
        if info.get('reached_goal', False):
            reward_dict['goal'] = goal_reward
        
        # Fuel reward (when fuel is collected) - sparse, medium value
        if current_fuel_collected > prev_fuel_collected:
            reward_dict['fuel'] = fuel_reward
        
        # Survival reward: step penalty + collision + revisit + death penalty
        # This captures all survival-related signals (navigation and staying alive)
        if info.get('died', False):
            reward_dict['survival'] = death_penalty
        else:
            # Base survival: step penalty (always present)
            # Collision and revisit penalties are also survival-related
            # We approximate by: total_reward - goal_reward - fuel_reward
            # This captures step_penalty + collision_penalty + revisit_penalty
            base_survival = reward - reward_dict['goal'] - reward_dict['fuel']
            reward_dict['survival'] = base_survival
        
        return reward_dict
    
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
            
            # Get current fuel collected
            current_fuel_collected = info.get('fuel_collected', 0)
            
            # Decompose reward based on environment state
            reward_dict = self._decompose_reward(reward, info, prev_fuel_collected, current_fuel_collected)
            
            # Track reward decomposition for monitoring
            for term in self.reward_terms:
                self.reward_decomposition_stats[term].append(reward_dict[term])
            
            # Get values from all critics
            with torch.no_grad():
                _, value_dict = self.network.forward(obs_tensor)
                value_dict_scalar = {k: v.item() for k, v in value_dict.items()}
            
            self.buffer.add(obs, action, reward_dict, value_dict_scalar, log_prob, done)
            
            ep_reward += reward
            ep_length += 1
            self.total_steps += 1
            
            if done:
                self.stats['episode_rewards'].append(ep_reward)
                self.stats['episode_lengths'].append(ep_length)
                self.stats['fuel_collected'].append(current_fuel_collected)
                self.stats['reached_goal'].append(1.0 if info.get('reached_goal') else 0.0)
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
            'mean_reward': np.mean(self.stats['episode_rewards']) if self.stats['episode_rewards'] else 0,
            'mean_length': np.mean(self.stats['episode_lengths']) if self.stats['episode_lengths'] else 0,
            'mean_fuel': np.mean(self.stats['fuel_collected']) if self.stats['fuel_collected'] else 0,
            'goal_rate': np.mean(self.stats['reached_goal']) if self.stats['reached_goal'] else 0,
            'death_rate': np.mean(self.stats['died']) if self.stats['died'] else 0,
        }
    
    def _compute_multi_gae(self):
        """Compute GAE for each reward term separately."""
        gamma, lam = self.config.gamma, self.config.gae_lambda
        
        self.returns = {}
        self.advantages = {}
        
        for term in self.reward_terms:
            rewards = np.array(self.buffer.rewards[term])
            values = np.array(self.buffer.values[term])
            dones = np.array(self.buffer.dones)
            
            n = len(rewards)
            advantages = np.zeros(n)
            last_gae = 0
            
            # Get last value
            last_value = 0 if dones[-1] else self.network.critics[term](
                torch.FloatTensor(self.buffer.observations[-1]).unsqueeze(0).to(self.device)
            ).item()
            
            for t in reversed(range(n)):
                next_value = last_value if t == n-1 else values[t+1]
                next_done = dones[-1] if t == n-1 else dones[t+1]
                delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
                advantages[t] = last_gae = delta + gamma * lam * (1 - next_done) * last_gae
            
            self.returns[term] = advantages + values
            # Normalize advantages following the paper (Eq. 8):
            # Ā_k_t = (A_k_t - E_t[A_k_t]) / sqrt(Var_t[A_k_t])
            # This ensures each objective contributes at the same scale
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 1e-8:
                # Standard normalization as per paper
                normalized_adv = (advantages - adv_mean) / adv_std
                # Light clipping to prevent extreme outliers (paper doesn't clip, but helps stability)
                self.advantages[term] = np.clip(normalized_adv, -5.0, 5.0)
            else:
                # If std is too small, just center around zero
                self.advantages[term] = advantages - adv_mean
    
    def update(self) -> Dict[str, float]:
        obs = torch.FloatTensor(np.array(self.buffer.observations)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        
        returns = {term: torch.FloatTensor(self.returns[term]).to(self.device) 
                  for term in self.reward_terms}
        advantages = {term: torch.FloatTensor(self.advantages[term]).to(self.device) 
                     for term in self.reward_terms}
        
        n = len(obs)
        total_loss = 0.0
        n_updates = 0
        
        for _ in range(self.config.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.config.batch_size):
                idx = indices[start:start + self.config.batch_size]
                
                new_log_probs, values_dict, entropy = self.network.evaluate_actions(obs[idx], actions[idx])
                
                # Combine advantages from all critics using weighted sum
                # Following the paper: each advantage is independently normalized,
                # then combined with weights. No need to normalize again after combination.
                # This matches Eq. 9 in the paper: L_π = E_t[Σ_k ω_k * Ā_k_t * log π]
                combined_advantages = torch.zeros_like(advantages[self.reward_terms[0]][idx])
                total_weight = sum(self.advantage_weights.get(term, 1.0) for term in self.reward_terms)
                
                for term in self.reward_terms:
                    weight = self.advantage_weights.get(term, 1.0)
                    # Each advantage is already normalized (Ā_k_t in the paper)
                    # Combine with weights: ω_k * Ā_k_t
                    combined_advantages += advantages[term][idx] * weight / total_weight
                
                # Note: According to the paper, we should NOT normalize again after combination
                # The paper uses PopArt for value normalization, but for advantage combination,
                # the weighted sum of already-normalized advantages is sufficient
                
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * combined_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                   1 + self.config.clip_epsilon) * combined_advantages
                
                # Actor loss: PPO clipped objective + entropy bonus
                policy_loss = -torch.min(surr1, surr2).mean()
                actor_loss = policy_loss - self.config.entropy_coef * entropy.mean()
                
                # Update Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.network.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update each Critic separately
                total_critic_loss = 0.0
                for term in self.reward_terms:
                    critic_loss = nn.functional.mse_loss(values_dict[term], returns[term][idx])
                    
                    self.critic_optimizers[term].zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.network.critics[term].parameters(), self.config.max_grad_norm)
                    self.critic_optimizers[term].step()
                    
                    total_critic_loss += critic_loss.item()
                
                total_loss += (actor_loss.item() + total_critic_loss)
                n_updates += 1
        
        return {'loss': total_loss / n_updates}
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, _, _ = self.network.get_action(obs_tensor, deterministic)
        return action
    
    def evaluate(self, n_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate the agent with DETERMINISTIC actions (no exploration).
        This gives the true performance of the learned policy.
        """
        results = {'goal': 0, 'death': 0, 'fuel': [], 'steps': [], 'reward': []}
        
        for _ in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0.0
            
            while not done:
                # Deterministic action (no exploration!)
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
    
    def save(self, path: str):
        torch.save({
            'network': self.network.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizers': {term: opt.state_dict() for term, opt in self.critic_optimizers.items()},
            'config': self.config,
            'reward_terms': self.reward_terms,
        }, path)
        print(f"Saved: {path}")
    
    def load(self, path: str):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(data['network'])
        if 'actor_optimizer' in data:
            self.actor_optimizer.load_state_dict(data['actor_optimizer'])
            for term in self.reward_terms:
                if term in data['critic_optimizers']:
                    self.critic_optimizers[term].load_state_dict(data['critic_optimizers'][term])
        print(f"Loaded: {path}")

