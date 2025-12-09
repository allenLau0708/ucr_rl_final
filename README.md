# Learning to Survive in a Dual-Objective Environment

PPO agents learn to survive in a dual-objective gridworld: reaching a goal while collecting scarce fuel. This repository contains the environment, single-critic and multi-critic PPO implementations, training utilities, evaluation scripts, and a polished Pygame visualizer.

---

## Contents

```
final/
├── agent.py                  # Single-critic PPO agent
├── agent_multi_critic.py     # Multi-critic PPO agent (separate critic per reward term)
├── config.py                 # PPO and environment configuration
├── environment.py            # FuelCollectionEnv gymnasium environment
├── train_vanilla_ppo.py                  # Training script for single-critic PPO
├── train_multi_critic.py     # Training script for multi-critic PPO
├── evaluate_models.py        # Evaluation and comparison script
├── visualize.py              # Interactive visualizer (Pygame)
├── best_models/              # Best trained models
│   ├── best_model_vanilla_ppo.pt
│   └── best_model_multi_critic.pt
├── models/                   # Training checkpoints (not tracked by git)
├── assets/icons/             # Custom PNG icons for visualizer
├── env_pool.npz              # Pre-generated environment pool for evaluation (not tracked by git)
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Setup

```bash
git clone <repository-url>
cd final

# Option A – Python's built-in venv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Option B – Conda environment
conda create -n fuel-rl python=3.10 -y
conda activate fuel-rl

pip install -r requirements.txt
```

---

## Usage

### Interactive Visualizer

Launch the Pygame visualizer to watch trained agents or play manually:

```bash
# Manual play
python visualize.py

# Watch a trained agent
python visualize.py --model best_models/best_model_vanilla_ppo.pt
python visualize.py --model best_models/best_model_multi_critic.pt
```

**Visualizer Features:**
- Toggle between global view and agent view (5×5 local view) with `V` key
- Live stats display (fuel, reward, trajectory)
- Custom PNG icons loaded from `assets/icons/` (falls back to emoji/ASCII if missing)
- Manual control with arrow keys when no model is loaded

### Training Agents

#### Single-Critic PPO

```bash
python train_vanilla_ppo.py
```

**Default settings:**
- `timesteps=50_000_000` (50M steps)
- `seed=42`
- Saves checkpoints to `models/`
- Best model saved based on evaluation goal rate

**Arguments:**
```bash
python train_vanilla_ppo.py --timesteps 10000000 --seed 123 --save-dir models
```

#### Multi-Critic PPO

```bash
python train_multi_critic.py
```

The multi-critic approach uses separate value networks for each reward term (goal, fuel, survival), which helps balance the dual objectives more effectively.

**Arguments:**
```bash
python train_multi_critic.py --timesteps 10000000 --seed 123 --save-dir models
```

**Training Output:**
- Model checkpoints saved to `models/` with timestamps
- Training history saved as JSON files
- TensorBoard logs in `runs/` directory
- Best model automatically saved based on evaluation performance

**Monitor Training:**
```bash
tensorboard --logdir=runs
# Then open http://localhost:6006
```

### Evaluating Models

Compare two trained models across a pool of environments:

```bash
python evaluate_models.py \
    --modelA best_models/best_model_vanilla_ppo.pt \
    --modelB best_models/best_model_multi_critic.pt \
    --env-pool env_pool.npz \
    --n-maps 100 \
    --out eval_results.json
```

This generates:
- Per-map statistics (goal rate, death rate, fuel collected, etc.)
- Aggregated comparison metrics
- CSV and JSON output files

---

## Environment Summary

**Grid:** 12×12 cells with randomly placed goal, obstacles, and fuel items each episode

**Observation Space (30-D):**
- Agent position (2): normalized x, y coordinates
- Goal direction (2): normalized direction vector to goal
- Current fuel (1): normalized remaining fuel ratio
- 5×5 egocentric map (25): local view around agent (empty/goal/obstacle/fuel)

**Action Space:** 4 discrete actions
- `0`: Move up
- `1`: Move down
- `2`: Move left
- `3`: Move right

**Rewards:**
- `+100.0`: Reaching goal
- `+5.0`: Collecting fuel item
- `-0.1`: Per step penalty
- `-1.0`: Collision with obstacle/wall
- `-0.5`: Revisiting a cell
- `-50.0`: Death penalty (fuel depletion)

**Challenge:**
- Initial fuel: 14 units
- Fuel per item: 10 units
- Goal typically requires 20+ steps
- Agent must collect fuel items to survive and reach the goal

The environment exposes standard Gymnasium API (`reset`, `step`, `render`), enabling easy integration with other RL algorithms.

---

## Architecture

### Single-Critic PPO (`agent.py`)
- Shared actor-critic network
- Single value function estimates total return
- Standard PPO with clipped surrogate objective

### Multi-Critic PPO (`agent_multi_critic.py`)
- Shared actor network
- Separate critic networks for each reward term:
  - Goal critic: estimates value for goal-reaching
  - Fuel critic: estimates value for fuel collection
  - Survival critic: estimates value for staying alive
- Combined advantage calculation from all critics
- Helps prevent exploitation of easy reward terms

---

## Custom Icons

The visualizer supports custom PNG icons:

1. Export transparent PNGs (≈512×512) for each role:
   - `agent.png`
   - `goal.png`
   - `fuel.png`
   - `fuel_collected.png`
   - `obstacle.png`
   - `unknown.png` (for fog of war)

2. Place them in `assets/icons/`

3. The visualizer automatically scales and uses them for both the main grid and the mini-map. Missing files fall back to emoji/ASCII characters.

---

## Project Structure

- **Training:** `train_vanilla_ppo.py` and `train_multi_critic.py` handle training loops, evaluation, and checkpointing
- **Agents:** `agent.py` (single-critic) and `agent_multi_critic.py` (multi-critic) implement PPO
- **Evaluation:** `evaluate_models.py` compares models across environment pools
- **Visualization:** `visualize.py` provides interactive Pygame interface
- **Configuration:** `config.py` contains hyperparameters and environment settings

---

## Tips & Experiments

- **Hyperparameter tuning:** Adjust `PPOConfig` in `config.py` or modify training scripts
- **Environment difficulty:** Modify `EnvConfig` in `environment.py` (grid size, fuel amounts, maze density)
- **Training duration:** Default is 50M steps; reduce `--timesteps` for faster experiments
- **Evaluation:** Use `evaluate_models.py` with `env_pool.npz` for fair model comparison
- **Visualization:** Launch TensorBoard to monitor training curves and metrics

---

## Troubleshooting

- **No emoji/PNG icons:** Ensure `pygame` finds system fonts or provide PNGs in `assets/icons/`
- **Black window on macOS:** Run `python -m pygame.examples.aliens` once to grant display permissions, or set `SDL_VIDEODRIVER=x11`
- **Slow training:** Reduce `total_timesteps`, shrink `hidden_sizes`, or lower `n_steps` in `PPOConfig`
- **Model loading errors:** Ensure model architecture matches (single-critic vs multi-critic)
- **CUDA errors:** Models default to CPU; modify `device` in `PPOConfig` if using GPU

---

## Results

Best models are saved in `best_models/`:
- `best_model_vanilla_ppo.pt`: Single-critic PPO agent
- `best_model_multi_critic.pt`: Multi-critic PPO agent

Training checkpoints and evaluation results are gitignored (see `.gitignore`).

---

## License & Attribution

Course project for CS258/EE227 (Fall 2025) by Zeli Liu & Hefeifei Jiang. 
