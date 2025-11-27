# Dual-Objective Reinforcement Learning: Learning to Survive

**CS258 Final Project**  
**Authors:** Zeli Liu, Hefeifei Jiang  
**Date:** November 2025

---

## Project Overview

This project implements a **Dual-Objective Reinforcement Learning** system where an agent must learn to balance two competing goals:

1. **Efficient Navigation** - Reaching a designated goal position as quickly as possible
2. **Resource Collection** - Gathering reward items scattered throughout the environment

Rather than optimizing for a single task, the agent must learn policies that find optimal trade-offs between these two distinct objectives.

### Key Features

- 🎮 **Lightweight 2D Gridworld Environment** - A simple yet effective testbed for multi-objective RL
- 🤖 **PPO Implementation** - Proximal Policy Optimization adapted for dual-objective settings
- ⚖️ **Multiple Reward Schemes** - Linear Scalarization and Reward Scheduling
- 📊 **Comprehensive Analysis** - Visualization tools for understanding agent behavior and objective trade-offs

---

## Project Structure

```
final/
├── main.py              # Main entry point with demos and quick training
├── config.py            # Configuration settings for all components
├── environment.py       # 2D Gridworld environment implementation
├── agent.py            # PPO agent implementation
├── reward_schemes.py   # Reward scalarization techniques
├── train.py            # Training script with experiment comparison
├── evaluate.py         # Evaluation and analysis tools
├── visualize.py        # Visualization utilities
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Installation

1. **Clone or download the project**

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Environment Demo
See how the gridworld environment works with random actions:
```bash
python main.py --demo env
```

### 2. Reward Schemes Demo
Understand how different reward scalarization methods work:
```bash
python main.py --demo rewards
```

### 3. Quick Training
Train an agent with specific reward settings:
```bash
# Navigation-focused agent (linear scalarization)
python main.py --train --timesteps 30000 --scheme linear --nav-weight 0.7

# Balanced agent (scheduling)
python main.py --train --timesteps 30000 --scheme scheduling
```

### 4. Full Comparison Experiments
Run comprehensive experiments comparing all reward schemes:
```bash
python main.py --compare
```

Or use the detailed training script:
```bash
python train.py --mode compare --timesteps 50000
```

### 5. Evaluate a Trained Model
```bash
python main.py --evaluate models/quick_train_linear.pt
```

---

## The Environment

### Grid Layout
- **White cells**: Empty space (navigable)
- **Green cell (G)**: Goal position
- **Yellow cells (*)**: Collectible items
- **Gray cells (#)**: Obstacles

### State Space
The observation is an 18-dimensional vector encoding:
- Agent position (normalized)
- Direction and distance to goal
- Direction and distance to nearest item
- Items collected ratio
- Remaining steps ratio
- Local obstacle map (3×3)

### Action Space
5 discrete actions:
- 0: Move Up
- 1: Move Down
- 2: Move Left
- 3: Move Right
- 4: Stay

### Rewards
- **Goal Reward**: +10.0 for reaching the goal
- **Item Reward**: +2.0 per item collected
- **Step Penalty**: -0.1 per step (encourages efficiency)
- **Obstacle Penalty**: -1.0 for hitting obstacles

---

## Reward Scalarization Schemes

### 1. Linear Scalarization
Combines objectives using fixed weights:
```
R_combined = w_nav × R_navigation + w_col × R_collection
```

**Properties:**
- Simple and interpretable
- Different weights produce different policies
- Can only find solutions on the convex Pareto front

### 2. Reward Scheduling
Dynamically adjusts weights during training:
```
w_nav(t) = w_initial + (w_final - w_initial) × progress(t)
```

**Schedule Types:**
- **Linear**: Uniform transition
- **Cosine**: Smooth, S-shaped transition
- **Exponential**: Slow start, fast finish
- **Step**: Discrete phase transitions

**Properties:**
- Can discover diverse solutions during training
- May find non-convex Pareto optimal solutions
- Enables curriculum learning approaches

---

## PPO Implementation

The agent uses **Proximal Policy Optimization** with:

- **Actor-Critic Architecture**: Shared feature layers with separate policy and value heads
- **GAE (Generalized Advantage Estimation)**: For stable advantage computation
- **Clipped Surrogate Objective**: Prevents large policy updates
- **Entropy Bonus**: Encourages exploration

### Hyperparameters (defaults)
| Parameter | Value |
|-----------|-------|
| Learning Rate | 3e-4 |
| Discount (γ) | 0.99 |
| GAE λ | 0.95 |
| Clip ε | 0.2 |
| Hidden Layers | [64, 64] |
| Batch Size | 64 |
| Epochs per Update | 10 |

---

## Expected Results

Different reward configurations lead to distinct agent behaviors:

### Navigation-Focused (nav_weight=0.8)
- High goal achievement rate (>80%)
- Fewer items collected
- Shorter episode lengths
- Policy: "Rush to goal, ignore items"

### Collection-Focused (nav_weight=0.2)
- Lower goal achievement rate
- More items collected per episode
- Longer episode lengths
- Policy: "Collect everything, goal is secondary"

### Balanced / Scheduled
- Moderate goal rate and item collection
- Finds trade-off solutions
- Scheduling may discover better overall policies

---

## Analysis and Visualization

The project includes comprehensive visualization tools:

### Training Curves
- Reward evolution (total, navigation, collection)
- Success metrics (goal rate, items collected)
- Learning dynamics (policy loss, entropy)

### Pareto Front Analysis
- Visualizes trade-offs between objectives
- Identifies Pareto optimal policies

### Policy Analysis
- Action distributions
- Value function heatmaps
- Goal-directed behavior assessment
- Obstacle avoidance patterns

### Custom Icons
- Place PNG icons inside `assets/icons/` named `agent.png`, `goal.png`, `fuel.png`, `fuel_collected.png`, `obstacle.png`, `unknown.png`
- Export vector artwork to transparent PNGs (≈512×512) so Pygame can scale them cleanly
- Any missing file automatically falls back to emoji or ASCII symbols

### Generate Visualizations
```bash
# Generate schedule comparison plot
python visualize.py --plot-schedules --output-dir visualizations/

# Generate training report for an experiment
python visualize.py --experiment-dir results/experiment_name/ --output-dir visualizations/
```

---

## Configuration Options

### Environment
```python
EnvironmentConfig(
    grid_size=10,      # Grid dimensions (10×10)
    num_items=5,       # Number of collectible items
    num_obstacles=8,   # Number of obstacles
    max_steps=100,     # Maximum steps per episode
    goal_reward=10.0,  # Reward for reaching goal
    item_reward=2.0,   # Reward per item collected
)
```

### PPO Agent
```python
PPOConfig(
    hidden_sizes=[64, 64],
    learning_rate=3e-4,
    gamma=0.99,
    clip_epsilon=0.2,
    n_steps=2048,
    total_timesteps=100000,
)
```

### Reward Scheme
```python
RewardSchemeConfig(
    scheme="linear",  # or "scheduling"
    navigation_weight=0.5,
    collection_weight=0.5,
    # For scheduling:
    initial_navigation_weight=0.3,
    final_navigation_weight=0.7,
    schedule_type="cosine",
)
```

---

## Experiments to Try

1. **Weight Sensitivity Study**
   - Train agents with nav_weight = {0.2, 0.4, 0.5, 0.6, 0.8}
   - Compare final performance and behaviors

2. **Schedule Type Comparison**
   - Compare linear, cosine, exponential, and step schedules
   - Analyze convergence speed and final performance

3. **Environment Complexity**
   - Vary grid_size, num_items, num_obstacles
   - Study how environment complexity affects learning

4. **Hyperparameter Tuning**
   - Experiment with learning rate, hidden sizes, batch size
   - Find optimal configurations for each reward scheme

---

## Troubleshooting

**Issue: Training is very slow**
- Reduce `total_timesteps` for quick experiments
- Use smaller `hidden_sizes` (e.g., [32, 32])
- Reduce `n_steps` to decrease memory usage

**Issue: Agent doesn't learn**
- Increase `learning_rate` slightly
- Increase `entropy_coef` for more exploration
- Check if reward weights are too imbalanced

**Issue: Matplotlib errors**
- Ensure you have a working backend: `pip install PyQt5` or use `matplotlib.use('Agg')`

---

## References

1. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017).
2. Van Moffaert, K., & Nowé, A. "Multi-Objective Reinforcement Learning using Sets of Pareto Dominating Policies." JMLR (2014).
3. Roijers, D. M., et al. "A Survey of Multi-Objective Sequential Decision-Making." JAIR (2013).

---

## License

This project is for educational purposes as part of CS258 coursework.

---

## Acknowledgments

This project was developed as part of the CS258 Reinforcement Learning course final project. We thank the course instructors for their guidance and feedback.


