"""
PPO Agent for Fuel Collection Environment.
CS258 Final Project: Learning to Survive in a Dual-Objective Environment
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
    """Critic network - outputs state value."""
    
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


class ActorCriticNetwork(nn.Module):
    """Decoupled Actor-Critic: two independent networks."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: List[int] = [256, 256]):
        super().__init__()
        
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_sizes)
        self.critic = CriticNetwork(obs_dim, hidden_sizes)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            logits = self.actor(obs)
            value = self.critic(obs)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = torch.argmax(probs, dim=-1) if deterministic else dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits = self.actor(obs)
        values = self.critic(obs)
        dist = Categorical(torch.softmax(logits, dim=-1))
        return dist.log_prob(actions), values, dist.entropy()


class RolloutBuffer:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, obs, action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def __len__(self):
        return len(self.observations)


class PPOAgent:
    """PPO agent for fuel collection."""
    
    def __init__(self, env: gym.Env, config: PPOConfig, reward_scheme=None):
        self.env = env
        self.config = config
        
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device(config.device)
        
        self.network = ActorCriticNetwork(
            self.obs_dim, self.action_dim, config.hidden_sizes
        ).to(self.device)
        
        # Separate optimizers for Actor and Critic
        self.actor_optimizer = optim.Adam(
            self.network.actor.parameters(), 
            lr=config.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.network.critic.parameters(), 
            lr=config.learning_rate * 2  # Critic often benefits from higher LR
        )
        self.buffer = RolloutBuffer()
        
        self.total_steps = 0
        self.episode_count = 0
        self.stats = {
            'episode_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'fuel_collected': deque(maxlen=100),
            'reached_goal': deque(maxlen=100),
            'died': deque(maxlen=100),
        }
    
    def collect_rollouts(self, n_steps: int) -> Dict[str, float]:
        self.buffer.reset()
        obs, info = self.env.reset()
        
        ep_reward = 0.0
        ep_length = 0
        
        for _ in range(n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, value = self.network.get_action(obs_tensor)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            self.buffer.add(obs, action, reward, value, log_prob, done)
            
            ep_reward += reward
            ep_length += 1
            self.total_steps += 1
            
            if done:
                self.stats['episode_rewards'].append(ep_reward)
                self.stats['episode_lengths'].append(ep_length)
                self.stats['fuel_collected'].append(info.get('fuel_collected', 0))
                self.stats['reached_goal'].append(1.0 if info.get('reached_goal') else 0.0)
                self.stats['died'].append(1.0 if info.get('died') else 0.0)
                
                self.episode_count += 1
                obs, info = self.env.reset()
                ep_reward = 0.0
                ep_length = 0
            else:
                obs = next_obs
        
        self._compute_gae()
        
        return {
            'mean_reward': np.mean(self.stats['episode_rewards']) if self.stats['episode_rewards'] else 0,
            'mean_length': np.mean(self.stats['episode_lengths']) if self.stats['episode_lengths'] else 0,
            'mean_fuel': np.mean(self.stats['fuel_collected']) if self.stats['fuel_collected'] else 0,
            'goal_rate': np.mean(self.stats['reached_goal']) if self.stats['reached_goal'] else 0,
            'death_rate': np.mean(self.stats['died']) if self.stats['died'] else 0,
        }
    
    def _compute_gae(self):
        gamma, lam = self.config.gamma, self.config.gae_lambda
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones)
        
        n = len(rewards)
        advantages = np.zeros(n)
        last_gae = 0
        
        last_value = 0 if dones[-1] else self.network.forward(
            torch.FloatTensor(self.buffer.observations[-1]).unsqueeze(0).to(self.device)
        )[1].item()
        
        for t in reversed(range(n)):
            next_value = last_value if t == n-1 else values[t+1]
            next_done = dones[-1] if t == n-1 else dones[t+1]
            delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - next_done) * last_gae
        
        self.returns = advantages + values
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def update(self) -> Dict[str, float]:
        obs = torch.FloatTensor(np.array(self.buffer.observations)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        returns = torch.FloatTensor(self.returns).to(self.device)
        advantages = torch.FloatTensor(self.advantages).to(self.device)
        
        n = len(obs)
        total_loss = 0.0
        n_updates = 0
        
        for _ in range(self.config.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.config.batch_size):
                idx = indices[start:start + self.config.batch_size]
                
                new_log_probs, values, entropy = self.network.evaluate_actions(obs[idx], actions[idx])
                
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                   1 + self.config.clip_epsilon) * advantages[idx]
                
                # Actor loss: PPO clipped objective + entropy bonus
                policy_loss = -torch.min(surr1, surr2).mean()
                actor_loss = policy_loss - self.config.entropy_coef * entropy.mean()
                
                # Critic loss: MSE between predicted and actual returns
                critic_loss = nn.functional.mse_loss(values, returns[idx])
                
                # Update Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.network.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.network.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()
                
                total_loss += (actor_loss.item() + critic_loss.item())
                n_updates += 1
        
        return {'loss': total_loss / n_updates}
    
    def train(self, total_timesteps: int) -> Dict[str, List]:
        history = {k: [] for k in ['timesteps', 'episode_rewards', 'episode_lengths', 
                                   'fuel_collected', 'goal_rate', 'death_rate']}
        
        n_updates = total_timesteps // self.config.n_steps
        
        for update in range(n_updates):
            stats = self.collect_rollouts(self.config.n_steps)
            self.update()
            
            history['timesteps'].append(self.total_steps)
            history['episode_rewards'].append(stats['mean_reward'])
            history['episode_lengths'].append(stats['mean_length'])
            history['fuel_collected'].append(stats['mean_fuel'])
            history['goal_rate'].append(stats['goal_rate'])
            history['death_rate'].append(stats['death_rate'])
            
            if (update + 1) % 10 == 0:
                print(f"Step {self.total_steps:6d} | "
                      f"Reward: {stats['mean_reward']:6.1f} | "
                      f"Goal: {stats['goal_rate']:5.1%} | "
                      f"Death: {stats['death_rate']:5.1%} | "
                      f"Fuel: {stats['mean_fuel']:.1f}")
        
        return history
    
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
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config,
        }, path)
        print(f"Saved: {path}")
    
    def load(self, path: str):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(data['network'])
        if 'actor_optimizer' in data:
            self.actor_optimizer.load_state_dict(data['actor_optimizer'])
            self.critic_optimizer.load_state_dict(data['critic_optimizer'])
        print(f"Loaded: {path}")
