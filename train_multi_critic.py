"""
Training Script for Fuel Collection Environment with Multi-Critic PPO.
CS258 Final Project: Learning to Survive in a Dual-Objective Environment

Multi-Objective RL: Separate critic per reward term to avoid exploiting easy terms.
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from config import PPOConfig
from environment import FuelCollectionEnv
from agent_multi_critic import MultiCriticPPOAgent


def train(timesteps: int = 50000000, seed: int = 42, save_dir: str = "models"):
    """Train the multi-critic agent."""
    print("="*60)
    print("  Training Fuel Collection Agent (Multi-Critic PPO)")
    print("="*60)
    
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    # TensorBoard for real-time monitoring
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"fuel_agent_multi_critic_{timestamp}")
    writer = SummaryWriter(log_dir)
    print(f"\n📊 TensorBoard: tensorboard --logdir=runs")
    print(f"   Then open: http://localhost:6006\n")
    
    env = FuelCollectionEnv()
    
    config = PPOConfig(
        hidden_sizes=[256, 256],      # Larger network for harder maze
        learning_rate=1e-4,           # Lower LR for stability
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.05,            # More exploration
        n_steps=2048,                 # 2048 / 128 = 16 exact mini-batches
        batch_size=128,               # Larger batch for stable gradients
        n_epochs=4,                   # Fewer epochs to prevent overfitting
        total_timesteps=timesteps,
    )
    
    agent = MultiCriticPPOAgent(env, config, reward_scheme=None)
    
    print(f"\nEnvironment: {env.grid_size}x{env.grid_size} grid")
    print(f"Initial fuel: {env.config.initial_fuel}")
    print(f"Observation dim: {env.observation_space.shape[0]}")
    print(f"Training: {timesteps:,} steps ({timesteps // config.n_steps} updates)")
    print(f"Multi-Critic: {len(agent.reward_terms)} critics ({', '.join(agent.reward_terms)})")
    print()
    
    history = {k: [] for k in ['timesteps', 'episode_rewards', 'episode_lengths', 
                               'fuel_collected', 'goal_rate', 'death_rate',
                               'eval_goal_rate', 'eval_death_rate']}
    
    n_updates = timesteps // config.n_steps
    best_eval_goal_rate = 0.0
    print_interval = 10   # Print every 10 updates (~20k steps)
    eval_interval = 10    # Eval every 10 updates
    
    # Log advantage weights for monitoring
    print(f"Advantage weights: {agent.advantage_weights}")
    critic_lr = config.learning_rate * 1.5
    print(f"Critic learning rates: goal={critic_lr:.2e}, "
          f"fuel={critic_lr:.2e}, "
          f"survival={critic_lr:.2e}")
    print()
    
    try:
        for update in range(n_updates):
            stats = agent.collect_rollouts(config.n_steps)
            agent.update()
            
            history['timesteps'].append(agent.total_steps)
            history['episode_rewards'].append(stats['mean_reward'])
            history['episode_lengths'].append(stats['mean_length'])
            history['fuel_collected'].append(stats['mean_fuel'])
            history['goal_rate'].append(stats['goal_rate'])
            history['death_rate'].append(stats['death_rate'])
            
            # Log to TensorBoard
            writer.add_scalar('Train/Reward', stats['mean_reward'], agent.total_steps)
            writer.add_scalar('Train/Goal_Rate', stats['goal_rate'], agent.total_steps)
            writer.add_scalar('Train/Death_Rate', stats['death_rate'], agent.total_steps)
            writer.add_scalar('Train/Fuel_Collected', stats['mean_fuel'], agent.total_steps)
            writer.add_scalar('Train/Episode_Length', stats['mean_length'], agent.total_steps)
            
            # Log individual critic advantages (if available)
            # This helps monitor if one critic dominates
            for term in agent.reward_terms:
                if hasattr(agent, 'advantages') and term in agent.advantages:
                    mean_adv = np.mean(agent.advantages[term])
                    std_adv = np.std(agent.advantages[term])
                    writer.add_scalar(f'Critic/{term}_advantage_mean', mean_adv, agent.total_steps)
                    writer.add_scalar(f'Critic/{term}_advantage_std', std_adv, agent.total_steps)
            
            # Log reward decomposition statistics
            for term in agent.reward_terms:
                if term in agent.reward_decomposition_stats and agent.reward_decomposition_stats[term]:
                    mean_reward = np.mean(agent.reward_decomposition_stats[term])
                    writer.add_scalar(f'Reward/{term}_mean', mean_reward, agent.total_steps)
            
            # Print training stats
            if (update + 1) % print_interval == 0 or update == 0:
                progress = (update + 1) / n_updates * 100
                print(f"[{progress:5.1f}%] Step {agent.total_steps:7d} | "
                      f"Reward: {stats['mean_reward']:6.1f} | "
                      f"Goal: {stats['goal_rate']:5.1%} (train) | "
                      f"Fuel: {stats['mean_fuel']:.1f}")
            
            # Periodic evaluation with deterministic policy
            if (update + 1) % eval_interval == 0:
                eval_stats = agent.evaluate(n_episodes=50)
                history['eval_goal_rate'].append(eval_stats['goal_rate'])
                history['eval_death_rate'].append(eval_stats['death_rate'])
                
                print(f"         EVAL: Goal={eval_stats['goal_rate']:.1%}, "
                      f"Death={eval_stats['death_rate']:.1%}, "
                      f"Fuel={eval_stats['mean_fuel']:.1f}")
                
                # Log eval to TensorBoard
                writer.add_scalar('Eval/Goal_Rate', eval_stats['goal_rate'], agent.total_steps)
                writer.add_scalar('Eval/Death_Rate', eval_stats['death_rate'], agent.total_steps)
                writer.add_scalar('Eval/Fuel_Collected', eval_stats['mean_fuel'], agent.total_steps)
                
                # Save best model based on EVALUATION (not training)
                if eval_stats['goal_rate'] > best_eval_goal_rate:
                    best_eval_goal_rate = eval_stats['goal_rate']
                    agent.save(os.path.join(save_dir, "best_model_multi_critic.pt"))
    
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("Training interrupted by user")
        print(f"Completed {update + 1}/{n_updates} updates ({agent.total_steps:,} steps)")
        print("="*60)
    except Exception as e:
        print("\n" + "="*60)
        print(f"Training error at update {update + 1}/{n_updates}")
        print(f"Steps completed: {agent.total_steps:,}")
        print(f"Error: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        # Save current model before exiting
        emergency_path = os.path.join(save_dir, f"emergency_save_{timestamp}.pt")
        agent.save(emergency_path)
        print(f"Emergency save: {emergency_path}")
        raise
    
    # Rename best_model.pt to agent_时间.pt
    temp_best_path = os.path.join(save_dir, "best_model_multi_critic.pt")
    model_path = os.path.join(save_dir, f"agent_multi_critic_{timestamp}.pt")
    
    print("\n" + "-"*60)
    if os.path.exists(temp_best_path):
        print("Loading best model for final evaluation...")
        agent.load(temp_best_path)
        os.rename(temp_best_path, model_path)
    else:
        print("No best model found, saving current model...")
        agent.save(model_path)
    
    # Save history
    history_path = os.path.join(save_dir, f"history_multi_critic_{timestamp}.json")
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)
    
    print("Final Evaluation (100 episodes, deterministic)...")
    final_eval = agent.evaluate(n_episodes=100)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total steps: {agent.total_steps:,}")
    print(f"Total episodes: ~{agent.episode_count}")
    print(f"Model saved: {model_path}")
    print(f"Best eval goal rate: {best_eval_goal_rate:.1%}")
    print()
    print("Final Evaluation Results:")
    print(f"  Goal Rate:  {final_eval['goal_rate']:.1%}")
    print(f"  Death Rate: {final_eval['death_rate']:.1%}")
    print(f"  Mean Fuel:  {final_eval['mean_fuel']:.1f}")
    print(f"  Mean Steps: {final_eval['mean_steps']:.1f}")
    
    # Log final results and close TensorBoard
    writer.add_hparams(
        {'timesteps': timesteps, 'seed': seed, 'method': 'multi_critic'},
        {'final/goal_rate': final_eval['goal_rate'],
         'final/death_rate': final_eval['death_rate']}
    )
    writer.close()
    
    return model_path, history


def main():
    parser = argparse.ArgumentParser(description='Train Fuel Collection Agent (Multi-Critic PPO)')
    parser.add_argument('--timesteps', type=int, default=50000000, help='Training timesteps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='models', help='Save directory')
    
    args = parser.parse_args()
    train(args.timesteps, args.seed, args.save_dir)


if __name__ == "__main__":
    main()

