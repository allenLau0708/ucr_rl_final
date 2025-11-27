# Fuel Collection RL

PPO agents learn to survive in a dual-objective gridworld: reaching a goal while collecting scarce fuel. This repository contains the environment, PPO implementation, training utilities, and a polished Pygame visualizer that supports custom PNG icons.

---

## Contents

```
final/
├── agent.py            # PPO actor-critic network and update logic
├── config.py           # Dataclasses describing all tunable hyperparameters
├── environment.py      # FuelCollectionEnv gymnasium environment
├── train.py            # Training/experiment runner
├── visualize.py        # Manual play + model playback (Pygame)
├── models/             # Saved checkpoints (.pt)
├── assets/icons/       # Optional icon PNGs used by the visualizer
├── requirements.txt    # Python dependencies
└── README.md
```

`main.py` has been removed—use `train.py` or `visualize.py` directly for all workflows.

---

## Setup

```bash
git clone https://github.com/allenLau0708/ucr_rl_final.git
cd ucr_rl_final

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

### Launch the interactive visualizer
```bash
python visualize.py                # Manual play
python visualize.py --model models/best_model_ppo_1126.pt   # Watch a trained agent
```

Features:
- Toggle between global/agent view (`V`)
- Live stats (fuel, reward, trajectory)
- Custom PNG icons loaded from `assets/icons/` (`agent.png`, `goal.png`, `fuel.png`, `fuel_collected.png`, `obstacle.png`, `unknown.png`)

### Train PPO agents
```bash
python train.py
```

Defaults (override via CLI flags):
- `total_timesteps=50_000`
- `scheme="linear"`
- `navigation_weight=0.6`, `collection_weight=0.4`
- `schedule_type="linear"` (when `scheme="scheduling"`)
- checkpoints saved to `models/` every evaluation interval

Other useful flags (`train.py -h`):
- `--scheme {linear,scheduling}`
- `--nav-weight`, `--col-weight`
- `--schedule-type {linear,cosine,exp,step}`
- `--eval-freq`, `--save-path`

Checkpoints are saved under `models/` and TensorBoard logs under `runs/`.

---

## Environment Summary

- **Grid:** 10×10 cells, random goal, obstacles, and fuel each episode
- **Observation:** 30-D vector (agent position, distances, remaining fuel ratio, 5×5 egocentric map)
- **Actions:** `0` up, `1` down, `2` left, `3` right
- **Rewards:** +10 goal, +2 per fuel, -0.1 per step, -1 obstacle collision, terminal penalty on fuel depletion

The environment exposes standard Gymnasium API (`reset`, `step`, `render`), enabling easy integration with other RL algorithms.

---

## Custom Icons

1. Export transparent PNGs (≈512×512) for each role: `agent`, `goal`, `fuel`, `fuel_collected`, `obstacle`, `unknown`.
2. Place them in `assets/icons/`.
3. The visualizer automatically scales and uses them for both the main grid and the mini-map. Missing files fall back to emoji/ASCII characters.

---

## Tips & Experiments

- **Trade-off sweeps:** run `train.py` with `--nav-weight` ∈ {0.2, 0.5, 0.8} to map the Pareto front between goal-reaching and collection.
- **Scheduling curricula:** `--scheme scheduling --schedule-type step --initial-nav-weight 0.3 --final-nav-weight 0.9`.
- **Ablations:** tweak `config.py` (grid size, fuel budget, reward magnitudes) to stress-test policies.
- **Visualization of logs:** launch TensorBoard on `runs/` for reward curves and diagnostics.

---

## Troubleshooting

- **No emoji/PNG icons:** ensure `pygame` finds system fonts or provide PNGs as described above.
- **Black window on macOS:** run `python -m pygame.examples.aliens` once to grant display permissions, or set `SDL_VIDEODRIVER=x11`.
- **Slow training:** reduce `total_timesteps`, shrink `hidden_sizes`, or lower `n_steps` in `PPOConfig`.

---

## License & Attribution

Course project for CS258 (Fall 2025) by Zeli Liu & Hefeifei Jiang. Feel free to fork for educational or research purposes. Please cite the original PPO paper (Schulman et al., 2017) if you use the agent implementation.
