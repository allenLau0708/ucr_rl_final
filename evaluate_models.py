"""
Evaluate two saved models (single-critic PPO vs multi-critic PPO) across N maps.

Usage:
    python evaluate_models.py --modelA models/best_model_ppo_1126.pt \
        --modelB models/best_model_multi_critic.pt --env-pool env_pool.npz --n-maps 100 --out results.json

The script will load the environment pool, run one deterministic episode per model per map,
collect metrics and write a JSON with per-map and aggregated statistics.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
import os
from typing import Tuple

import numpy as np
import torch

from environment import FuelCollectionEnv
from agent import ActorCriticNetwork
from agent_multi_critic import MultiCriticNetwork
from config import PPOConfig


def load_state_dict_robust(network, path: str):
    """Load state_dict from file into network with simple remapping fallbacks.
    Returns (loaded, info_dict)
    info_dict contains keys: remapped(bool), missing_count, unexpected_count
    """
    data = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = data.get('network', data)

    model_keys = set(network.state_dict().keys())
    saved_keys = set(state_dict.keys())

    info = {'remapped': False, 'missing_count': 0, 'unexpected_count': 0}

    # direct match
    if model_keys == saved_keys:
        network.load_state_dict(state_dict)
        return True, info

    # try remapping '.network.' <-> '.net.'
    mapped = None
    if any('.network.' in k for k in saved_keys) and any('.net.' in k for k in model_keys):
        mapped = {k.replace('.network.', '.net.'): v for k, v in state_dict.items()}
        info['remapped'] = True
    elif any('.net.' in k for k in saved_keys) and any('.network.' in k for k in model_keys):
        mapped = {k.replace('.net.', '.network.'): v for k, v in state_dict.items()}
        info['remapped'] = True

    try:
        if mapped is not None:
            network.load_state_dict(mapped, strict=False)
            provided_keys = set(mapped.keys())
        else:
            network.load_state_dict(state_dict, strict=False)
            provided_keys = set(state_dict.keys())

        missing = model_keys - provided_keys
        unexpected = provided_keys - model_keys
        info['missing_count'] = len(missing)
        info['unexpected_count'] = len(unexpected)
        return True, info
    except Exception as e:
        print(f"Failed to load state dict from {path}: {e}")
        return False, info


def build_network_from_file(path: str) -> Tuple[object, dict]:
    # Load full checkpoint (allow unpickling of config objects)
    data = torch.load(path, map_location='cpu', weights_only=False)
    config = data.get('config', PPOConfig())
    state_dict = data.get('network', data)
    is_multi = any('critics.' in k for k in state_dict.keys())

    if is_multi:
        reward_terms = data.get('reward_terms', ['goal', 'fuel', 'survival'])
        net = MultiCriticNetwork(obs_dim=30, action_dim=4, hidden_sizes=config.hidden_sizes, reward_terms=reward_terms)
    else:
        net = ActorCriticNetwork(obs_dim=30, action_dim=4, hidden_sizes=config.hidden_sizes)

    loaded, info = load_state_dict_robust(net, path)
    if not loaded:
        raise RuntimeError(f"Failed to load model from {path}")
    net.eval()

    # Make config JSON-serializable
    try:
        from dataclasses import is_dataclass, asdict
        if is_dataclass(config):
            config_serial = asdict(config)
        else:
            config_serial = str(config)
    except Exception:
        config_serial = str(config)

    return net, {'is_multi': is_multi, 'config': config_serial, **info}


def set_env_to_pool_index(env: FuelCollectionEnv, idx: int):
    # set internal grid and fuel_items from env_pool
    env_data = env.env_pool[idx]
    env.grid = env_data['grid'].copy()
    env.fuel_items = [list(f) for f in env_data['fuel_items']]
    env.agent_pos = [0, 0]
    env.goal_pos = [env.grid_size - 1, env.grid_size - 1]
    env.current_fuel = env.config.initial_fuel
    env.steps_taken = 0
    env.fuel_collected = 0
    env.total_reward = 0.0
    env.visited = {tuple(env.agent_pos)}


def run_episode_with_network(env: FuelCollectionEnv, net, deterministic=True) -> dict:
    obs = env._get_observation()
    info = env._get_info()
    done = False
    while not done:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        # decide action depending on network API
        if hasattr(net, 'get_action'):
            a, *_ = net.get_action(obs_t, deterministic=deterministic)
            if isinstance(a, torch.Tensor):
                action = int(a.item())
            else:
                action = int(a)
        else:
            # fallback: forward to get logits and pick argmax
            logits = net.actor(obs_t) if hasattr(net, 'actor') else net(obs_t)[0]
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.argmax(probs, dim=-1).item())

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return {
        'reached_goal': bool(info.get('reached_goal', False)),
        'died': bool(info.get('died', False)),
        'fuel_collected': int(info.get('fuel_collected', 0)),
        'steps': int(info.get('steps_taken', 0)),
        'total_reward': float(info.get('total_reward', 0.0)),
    }


def evaluate_models(modelA: str, modelB: str, env_pool_path: str, n_maps: int = 100, seed: int = 42):
    env = FuelCollectionEnv()
    if os.path.exists(env_pool_path):
        env.load_env_pool(env_pool_path)
    else:
        raise FileNotFoundError(f"Env pool not found: {env_pool_path}")

    if n_maps > len(env.env_pool):
        raise ValueError(f"env_pool contains only {len(env.env_pool)} maps, requested {n_maps}")

    netA, metaA = build_network_from_file(modelA)
    netB, metaB = build_network_from_file(modelB)

    results = {'modelA': {'meta': metaA, 'per_map': []}, 'modelB': {'meta': metaB, 'per_map': []}}

    for i in range(n_maps):
        set_env_to_pool_index(env, i)
        resA = run_episode_with_network(env, netA, deterministic=True)

        set_env_to_pool_index(env, i)
        resB = run_episode_with_network(env, netB, deterministic=True)

        results['modelA']['per_map'].append(resA)
        results['modelB']['per_map'].append(resB)

    # aggregate
    def summarize(per_map_list):
        goals = sum(1 for r in per_map_list if r['reached_goal'])
        deaths = sum(1 for r in per_map_list if r['died'])
        mean_fuel = float(np.mean([r['fuel_collected'] for r in per_map_list]))
        mean_steps = float(np.mean([r['steps'] for r in per_map_list]))
        mean_reward = float(np.mean([r['total_reward'] for r in per_map_list]))
        return {
            'maps': len(per_map_list),
            'goal_rate': goals / max(1, len(per_map_list)),
            'death_rate': deaths / max(1, len(per_map_list)),
            'mean_fuel': mean_fuel,
            'mean_steps': mean_steps,
            'mean_reward': mean_reward,
        }

    results['modelA']['summary'] = summarize(results['modelA']['per_map'])
    results['modelB']['summary'] = summarize(results['modelB']['per_map'])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelA', required=True)
    parser.add_argument('--modelB', required=True)
    parser.add_argument('--env-pool', default='env_pool.npz')
    parser.add_argument('--n-maps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='eval_results.json')

    args = parser.parse_args()

    results = evaluate_models(args.modelA, args.modelB, args.env_pool, args.n_maps, args.seed)

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Wrote results to {args.out}")

    # Visualization / CSV output
    try:
        visualize_results(results, args.out)
    except Exception as e:
        print(f"Visualization failed: {e}")


def visualize_results(results: dict, out_path: str):
    """Generate CSV and PNG visualizations from evaluation results.

    Produces:
      - <out_path>.per_map.csv : per-map metrics for both models
      - <out_path>.summary.png : bar chart comparing summary metrics
      - <out_path>.reward_diff_hist.png : histogram of per-map reward differences (B - A)
      - <out_path>.reward_scatter.png : scatter plot of reward per map (A vs B)
    """
    import csv
    import os
    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise RuntimeError("matplotlib required for visualization; please pip install matplotlib")

    base = os.path.splitext(out_path)[0]

    perA = results['modelA']['per_map']
    perB = results['modelB']['per_map']
    n = len(perA)

    # Write per-map CSV
    csv_path = base + '.per_map.csv'
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        header = ['map_idx', 'A_reached_goal', 'A_died', 'A_fuel', 'A_steps', 'A_reward',
                  'B_reached_goal', 'B_died', 'B_fuel', 'B_steps', 'B_reward']
        writer.writerow(header)
        for i in range(n):
            a = perA[i]
            b = perB[i]
            row = [i, int(a['reached_goal']), int(a['died']), a['fuel_collected'], a['steps'], a['total_reward'],
                   int(b['reached_goal']), int(b['died']), b['fuel_collected'], b['steps'], b['total_reward']]
            writer.writerow(row)

    print(f"Wrote per-map CSV: {csv_path}")

    # Summary bar chart
    summaryA = results['modelA']['summary']
    summaryB = results['modelB']['summary']

    metrics = ['goal_rate', 'death_rate', 'mean_fuel', 'mean_steps', 'mean_reward']
    labels = ['Goal Rate', 'Death Rate', 'Fuel', 'Steps', 'Reward']
    valsA = [summaryA[m] for m in metrics]
    valsB = [summaryB[m] for m in metrics]

    x = list(range(len(metrics)))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([xi - w/2 for xi in x], valsA, width=w, label='Model A: Vanilla PPO')
    ax.bar([xi + w/2 for xi in x], valsB, width=w, label='Model B: Multi-Critic PPO')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title('Summary Comparison')
    ax.legend()
    plt.tight_layout()
    summary_png = base + '.summary.png'
    fig.savefig(summary_png)
    plt.close(fig)
    print(f"Wrote summary plot: {summary_png}")

    # # Reward difference histogram (B - A)
    # rewardA = np.array([r['total_reward'] for r in perA])
    # rewardB = np.array([r['total_reward'] for r in perB])
    # diff = rewardB - rewardA

    # fig, ax = plt.subplots(figsize=(6, 4))
    # ax.hist(diff, bins=20, color='C2', edgecolor='k')
    # ax.axvline(0, color='k', linestyle='--')
    # ax.set_title('Per-map Reward Difference (B - A)')
    # ax.set_xlabel('Reward Difference')
    # ax.set_ylabel('Count')
    # plt.tight_layout()
    # hist_png = base + '.reward_diff_hist.png'
    # fig.savefig(hist_png)
    # plt.close(fig)
    # print(f"Wrote reward diff histogram: {hist_png}")

    # # Scatter plot A vs B rewards
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.scatter(rewardA, rewardB, alpha=0.7)
    # mx = max(rewardA.max(), rewardB.max())
    # mn = min(rewardA.min(), rewardB.min())
    # ax.plot([mn, mx], [mn, mx], color='k', linestyle='--')
    # ax.set_xlabel('Model A Reward')
    # ax.set_ylabel('Model B Reward')
    # ax.set_title('Per-map Reward: Model A vs Model B')
    # plt.tight_layout()
    # scatter_png = base + '.reward_scatter.png'
    # fig.savefig(scatter_png)
    # plt.close(fig)
    # print(f"Wrote reward scatter: {scatter_png}")


if __name__ == '__main__':
    main()
