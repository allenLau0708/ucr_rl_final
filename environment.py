"""
Fuel Collection Environment
CS258 Final Project: Learning to Survive in a Dual-Objective Environment

OBJECTIVE: Reach the goal with limited fuel
CHALLENGE: Must collect fuel items along the way
OBSERVATION: Only 5x5 local view + goal direction + fuel level
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
from dataclasses import dataclass
from collections import deque


@dataclass 
class EnvConfig:
    """Environment configuration."""
    grid_size: int = 12
    num_fuel_items: int = 4
    
    initial_fuel: int = 14        # Not enough to reach goal directly
    fuel_per_item: int = 10       # Each fuel extends range
    
    goal_reward: float = 100.0
    fuel_pickup_reward: float = 5.0
    death_penalty: float = -50.0
    step_penalty: float = -0.1
    collision_penalty: float = -1.0   # Penalty for hitting wall/obstacle
    revisit_penalty: float = -0.5     # Penalty for revisiting a cell
    
    maze_density: float = 0.20    # 20% walls - more open map


class CellType:
    EMPTY = 0
    GOAL = 1
    OBSTACLE = 2
    FUEL = 3


class FuelCollectionEnv(gym.Env):
    """
    Fuel Collection Environment.
    
    Observation Space (28 dimensions):
    - Goal direction (2): normalized direction to goal
    - Current fuel (1): normalized fuel level
    - 5x5 local view (25): what agent can see around itself
    
    NO "cheating" information like:
    - Nearest fuel direction (agent must find fuel by exploring)
    - Number of remaining fuel items
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_NAMES = ['↑', '↓', '←', '→']
    
    def __init__(self, config: Optional[EnvConfig] = None, render_mode: str = None):
        super().__init__()
        
        self.config = config if config else EnvConfig()
        self.render_mode = render_mode
        self.grid_size = self.config.grid_size
        
        self.action_space = spaces.Discrete(4)
        
        # Observation: position(2) + goal_dir(2) + fuel(1) + 5x5_view(25) = 30
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(30,), dtype=np.float32
        )
        
        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.fuel_items = None
        self.current_fuel = 0
        self.steps_taken = 0
        self.fuel_collected = 0
        self.total_reward = 0.0
        self.visited = set()  # Track visited cells
        
        # Pre-generated environment pool
        self.env_pool = []
        self.pool_index = 0
    
    def generate_env_pool(self, num_envs: int = 1000, verbose: bool = True):
        """Pre-generate a pool of valid, interesting environments."""
        self.env_pool = []
        
        if verbose:
            print(f"Generating {num_envs} environments...")
        
        attempts = 0
        max_attempts = num_envs * 20  # Allow many attempts
        
        while len(self.env_pool) < num_envs and attempts < max_attempts:
            attempts += 1
            
            # Reset grid
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            self.agent_pos = [0, 0]
            self.goal_pos = [self.grid_size - 1, self.grid_size - 1]
            self.grid[self.goal_pos[0], self.goal_pos[1]] = CellType.GOAL
            
            # Generate maze
            self._create_maze_structure()
            self._place_fuel_strategic()
            
            # Validate
            if self._verify_solvable() and self._is_interesting():
                # Save this environment
                env_data = {
                    'grid': self.grid.copy(),
                    'fuel_items': [f.copy() for f in self.fuel_items]
                }
                self.env_pool.append(env_data)
                
                if verbose and len(self.env_pool) % 100 == 0:
                    print(f"  Generated {len(self.env_pool)}/{num_envs} environments...")
        
        if verbose:
            print(f"Done! Generated {len(self.env_pool)} valid environments.")
        
        # Shuffle the pool
        self.np_random.shuffle(self.env_pool)
        self.pool_index = 0
        
        return len(self.env_pool)
    
    def save_env_pool(self, path: str = "env_pool.npz"):
        """Save environment pool to file."""
        if not self.env_pool:
            print("No environment pool to save!")
            return
        
        grids = np.array([e['grid'] for e in self.env_pool])
        fuel_items = [e['fuel_items'] for e in self.env_pool]
        
        np.savez(path, grids=grids, fuel_items=np.array(fuel_items, dtype=object))
        print(f"Saved {len(self.env_pool)} environments to {path}")
    
    def load_env_pool(self, path: str = "env_pool.npz"):
        """Load environment pool from file."""
        data = np.load(path, allow_pickle=True)
        grids = data['grids']
        fuel_items = data['fuel_items']
        
        self.env_pool = []
        for i in range(len(grids)):
            self.env_pool.append({
                'grid': grids[i],
                'fuel_items': [list(f) for f in fuel_items[i]]
            })
        
        self.pool_index = 0
        print(f"Loaded {len(self.env_pool)} environments from {path}")
        return len(self.env_pool)
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.agent_pos = [0, 0]
        self.goal_pos = [self.grid_size - 1, self.grid_size - 1]
        
        # Auto-generate pool on first reset
        if not self.env_pool:
            self.generate_env_pool(num_envs=1000)
        
        # Random sample from pool
        idx = self.np_random.integers(0, len(self.env_pool))
        env_data = self.env_pool[idx]
        self.grid = env_data['grid'].copy()
        self.fuel_items = [f.copy() for f in env_data['fuel_items']]
        
        self.current_fuel = self.config.initial_fuel
        self.steps_taken = 0
        self.fuel_collected = 0
        self.total_reward = 0.0
        self.visited = {tuple(self.agent_pos)}  # Start position is visited
        
        return self._get_observation(), self._get_info()
    
    def _generate_maze(self):
        """Generate maze-like obstacles ensuring solvability."""
        max_attempts = 100
        best_map = None
        
        for attempt in range(max_attempts):
            # Reset grid
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            self.grid[self.goal_pos[0], self.goal_pos[1]] = CellType.GOAL
            
            # Generate maze structure
            self._create_maze_structure()
            
            # Place fuel in strategic locations
            self._place_fuel_strategic()
            
            # Verify solvability
            if self._verify_solvable():
                # Prefer maps where fuel collection is required
                if self._is_interesting():
                    return  # Perfect - solvable AND requires fuel
                elif best_map is None:
                    # Save as backup (solvable but maybe too easy)
                    best_map = (self.grid.copy(), [f.copy() for f in self.fuel_items])
        
        # Use best map found, or fallback
        if best_map is not None:
            self.grid, self.fuel_items = best_map
            return
        
        # Fallback: simple map if maze generation fails
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.grid[self.goal_pos[0], self.goal_pos[1]] = CellType.GOAL
        self._place_fuel_strategic()
    
    def _create_maze_structure(self):
        """Create maze-like walls using randomized structure."""
        # Protected zones around start and goal
        protected = set()
        for di in range(-2, 3):
            for dj in range(-2, 3):
                protected.add((di, dj))
                protected.add((self.grid_size - 1 + di, self.grid_size - 1 + dj))
        
        # Create corridor-like structure
        num_walls = int(self.grid_size * self.grid_size * self.config.maze_density)
        
        # Add some structured walls (horizontal and vertical segments)
        for _ in range(num_walls // 3):
            # Random wall segment
            start_i = self.np_random.integers(1, self.grid_size - 1)
            start_j = self.np_random.integers(1, self.grid_size - 1)
            length = self.np_random.integers(2, 5)
            horizontal = self.np_random.random() > 0.5
            
            for k in range(length):
                if horizontal:
                    pos = (start_i, min(start_j + k, self.grid_size - 1))
                else:
                    pos = (min(start_i + k, self.grid_size - 1), start_j)
                
                if pos not in protected and self.grid[pos[0], pos[1]] == CellType.EMPTY:
                    if pos != tuple(self.goal_pos):
                        self.grid[pos[0], pos[1]] = CellType.OBSTACLE
        
        # Add scattered walls
        for _ in range(num_walls // 2):
            i = self.np_random.integers(0, self.grid_size)
            j = self.np_random.integers(0, self.grid_size)
            
            if (i, j) not in protected and self.grid[i, j] == CellType.EMPTY:
                if [i, j] != self.goal_pos:
                    self.grid[i, j] = CellType.OBSTACLE
        
        # Ensure path exists - carve path if needed
        self._ensure_path_exists()
    
    def _ensure_path_exists(self):
        """Carve a guaranteed path from start to goal."""
        # BFS to check connectivity
        if self._bfs_path(self.agent_pos, self.goal_pos) is not None:
            return
        
        # Carve a winding path
        current = list(self.agent_pos)
        while current != self.goal_pos:
            self.grid[current[0], current[1]] = CellType.EMPTY if self.grid[current[0], current[1]] == CellType.OBSTACLE else self.grid[current[0], current[1]]
            
            # Move towards goal with some randomness
            if self.np_random.random() < 0.7:
                # Move towards goal
                if current[0] < self.goal_pos[0]:
                    current[0] += 1
                elif current[0] > self.goal_pos[0]:
                    current[0] -= 1
                elif current[1] < self.goal_pos[1]:
                    current[1] += 1
                else:
                    current[1] -= 1
            else:
                # Random perpendicular move
                if current[0] != self.goal_pos[0]:
                    current[1] = max(0, min(self.grid_size - 1, current[1] + self.np_random.choice([-1, 1])))
                else:
                    current[0] = max(0, min(self.grid_size - 1, current[0] + self.np_random.choice([-1, 1])))
    
    def _place_fuel_strategic(self):
        """Place fuel SPREAD ACROSS different regions of the map."""
        self.fuel_items = []
        
        # Divide map into regions (quadrants) to ensure spread
        half = self.grid_size // 2
        regions = [
            (0, half, 0, half),           # Top-left
            (0, half, half, self.grid_size),      # Top-right  
            (half, self.grid_size, 0, half),      # Bottom-left
            (half, self.grid_size, half, self.grid_size),  # Bottom-right
        ]
        
        # Find the main path to avoid placing fuel directly on it
        main_path = self._bfs_path(self.agent_pos, self.goal_pos)
        main_path_set = set(main_path) if main_path else set()
        
        # Collect candidates from each region
        region_candidates = []
        for r_idx, (r1, r2, c1, c2) in enumerate(regions):
            candidates = []
            for i in range(r1, r2):
                for j in range(c1, c2):
                    if self.grid[i, j] == CellType.EMPTY:
                        if [i, j] != self.agent_pos and [i, j] != self.goal_pos:
                            if (i, j) not in main_path_set:  # Not on direct path
                                candidates.append((i, j))
            region_candidates.append(candidates)
        
        # Place fuel: one from each region first, then fill remaining
        placed = 0
        region_order = list(range(4))
        self.np_random.shuffle(region_order)
        
        # First pass: one fuel per region
        for r_idx in region_order:
            if placed >= self.config.num_fuel_items:
                break
            candidates = region_candidates[r_idx]
            if candidates:
                self.np_random.shuffle(candidates)
                for pos in candidates:
                    if self._is_reachable(self.agent_pos, list(pos)):
                        # Check minimum distance from other fuel
                        too_close = any(
                            abs(pos[0] - f[0]) + abs(pos[1] - f[1]) < 3
                            for f in self.fuel_items
                        )
                        if not too_close:
                            self.grid[pos[0], pos[1]] = CellType.FUEL
                            self.fuel_items.append(list(pos))
                            placed += 1
                            break
        
        # Second pass: fill remaining anywhere (but spread out)
        all_empty = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == CellType.EMPTY:
                    if [i, j] != self.agent_pos and [i, j] != self.goal_pos:
                        all_empty.append((i, j))
        
        self.np_random.shuffle(all_empty)
        for pos in all_empty:
            if placed >= self.config.num_fuel_items:
                break
            # Ensure minimum distance from existing fuel
            too_close = any(
                abs(pos[0] - f[0]) + abs(pos[1] - f[1]) < 3
                for f in self.fuel_items
            )
            if not too_close and self._is_reachable(self.agent_pos, list(pos)):
                self.grid[pos[0], pos[1]] = CellType.FUEL
                self.fuel_items.append(list(pos))
                placed += 1
        
        # Fallback: if still not enough, place anywhere reachable
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if placed >= self.config.num_fuel_items:
                    break
                if self.grid[i, j] == CellType.EMPTY and [i, j] != self.agent_pos:
                    if self._is_reachable(self.agent_pos, [i, j]):
                        self.grid[i, j] = CellType.FUEL
                        self.fuel_items.append([i, j])
                        placed += 1
    
    def _find_dead_ends(self):
        """Find cells with only one open neighbor (dead-ends in maze)."""
        dead_ends = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == CellType.EMPTY:
                    open_neighbors = 0
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            if self.grid[ni, nj] != CellType.OBSTACLE:
                                open_neighbors += 1
                    if open_neighbors == 1 and [i, j] != self.agent_pos:
                        dead_ends.append((i, j))
        
        return dead_ends
    
    def _bfs_path(self, start, goal):
        """BFS to find shortest path. Returns path as list of tuples or None."""
        queue = deque([(tuple(start), [tuple(start)])])
        visited = {tuple(start)}
        
        while queue:
            (i, j), path = queue.popleft()
            
            if [i, j] == goal:
                return path
            
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                    if (ni, nj) not in visited and self.grid[ni, nj] != CellType.OBSTACLE:
                        visited.add((ni, nj))
                        queue.append(((ni, nj), path + [(ni, nj)]))
        
        return None
    
    def _is_reachable(self, start, goal):
        """Check if goal is reachable from start."""
        return self._bfs_path(start, goal) is not None
    
    def _verify_solvable(self):
        """
        Verify the map is solvable using BFS with fuel state.
        Returns True only if agent can reach goal by collecting fuel.
        """
        # Basic check: path to goal must exist
        path_to_goal = self._bfs_path(self.agent_pos, self.goal_pos)
        if path_to_goal is None:
            return False
        
        direct_distance = len(path_to_goal) - 1
        
        # Must have fuel items
        if len(self.fuel_items) < 1:
            return False
        
        # BFS with state = (position, fuel_collected_bitmask)
        # This finds if we can reach goal with enough fuel
        from itertools import combinations
        
        fuel_positions = [tuple(f) for f in self.fuel_items]
        
        # Try to find a valid path: start -> collect some fuel -> goal
        # Use BFS where state = (pos, remaining_fuel, collected_fuel_set)
        
        initial_fuel = self.config.initial_fuel
        fuel_per_item = self.config.fuel_per_item
        
        # State: (row, col, fuel_remaining, frozenset of collected fuel indices)
        start_state = (self.agent_pos[0], self.agent_pos[1], initial_fuel, frozenset())
        
        queue = deque([start_state])
        visited = {(self.agent_pos[0], self.agent_pos[1], frozenset())}
        
        while queue:
            r, c, fuel, collected = queue.popleft()
            
            # Check if at goal
            if [r, c] == self.goal_pos:
                return True  # Solvable!
            
            # No fuel left, can't move
            if fuel <= 0:
                continue
            
            # Try all 4 directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                
                if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                    continue
                if self.grid[nr, nc] == CellType.OBSTACLE:
                    continue
                
                new_fuel = fuel - 1
                new_collected = collected
                
                # Check if stepping on fuel
                if (nr, nc) in fuel_positions:
                    fuel_idx = fuel_positions.index((nr, nc))
                    if fuel_idx not in collected:
                        new_fuel += fuel_per_item
                        new_collected = collected | {fuel_idx}
                
                # Skip if already visited this state (position + collected set)
                state_key = (nr, nc, new_collected)
                if state_key in visited:
                    continue
                
                visited.add(state_key)
                queue.append((nr, nc, new_fuel, new_collected))
        
        return False  # No valid path found
    
    def _is_interesting(self):
        """Check if the map requires collecting fuel (not trivially solvable)."""
        path_to_goal = self._bfs_path(self.agent_pos, self.goal_pos)
        if path_to_goal is None:
            return False
        
        direct_distance = len(path_to_goal) - 1
        
        # Map is interesting if direct path requires more fuel than we start with
        return direct_distance > self.config.initial_fuel
    
    def step(self, action: int):
        if self.current_fuel <= 0:
            info = self._get_info()
            info['died'] = True
            info['reached_goal'] = False
            return self._get_observation(), self.config.death_penalty, True, False, info
        
        self.current_fuel -= 1
        self.steps_taken += 1
        
        dx, dy = self.ACTIONS[action]
        new_pos = [self.agent_pos[0] + dx, self.agent_pos[1] + dy]
        
        reward = self.config.step_penalty
        
        # Check for collision
        hit_wall = not self._is_valid_position(new_pos)
        hit_obstacle = self._is_valid_position(new_pos) and self.grid[new_pos[0], new_pos[1]] == CellType.OBSTACLE
        
        if hit_wall or hit_obstacle:
            # Collision! Stay in place and get penalty
            reward += self.config.collision_penalty
        else:
            # Valid move
            self.agent_pos = new_pos
            pos_tuple = tuple(new_pos)
            
            # Check if revisiting
            if pos_tuple in self.visited:
                reward += self.config.revisit_penalty
            else:
                self.visited.add(pos_tuple)
            
            if self.grid[new_pos[0], new_pos[1]] == CellType.FUEL:
                self.current_fuel += self.config.fuel_per_item
                self.fuel_collected += 1
                reward += self.config.fuel_pickup_reward
                self.grid[new_pos[0], new_pos[1]] = CellType.EMPTY
                if new_pos in self.fuel_items:
                    self.fuel_items.remove(new_pos)
        
        terminated = False
        reached_goal = False
        died = False
        
        if self.agent_pos == self.goal_pos:
            reached_goal = True
            terminated = True
            reward += self.config.goal_reward
        
        if self.current_fuel <= 0 and not reached_goal:
            died = True
            terminated = True
            reward += self.config.death_penalty
        
        self.total_reward += reward
        
        info = self._get_info()
        info['died'] = died
        info['reached_goal'] = reached_goal
        
        return self._get_observation(), reward, terminated, False, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get observation:
        
        - Position (2): agent's absolute position (normalized)
        - Goal direction (2): which direction goal is
        - Fuel level (1): current fuel
        - 5x5 local view (25): what agent can see
        
        Total: 30 dimensions
        """
        obs = np.zeros(30, dtype=np.float32)
        
        # Agent's absolute position (normalized to [-1, 1])
        obs[0] = self.agent_pos[0] / (self.grid_size - 1) * 2 - 1
        obs[1] = self.agent_pos[1] / (self.grid_size - 1) * 2 - 1
        
        # Goal direction (normalized)
        goal_dist = max(1, self._manhattan_distance(self.agent_pos, self.goal_pos))
        obs[2] = (self.goal_pos[0] - self.agent_pos[0]) / goal_dist
        obs[3] = (self.goal_pos[1] - self.agent_pos[1]) / goal_dist
        
        # Current fuel level (normalized)
        obs[4] = min(1.0, self.current_fuel / 30.0) * 2 - 1  # [-1, 1]
        
        # 5x5 local view (what agent can actually see)
        # Different codes: WALL=-1.0, OBSTACLE=-0.5, EMPTY=0, FUEL=0.5, GOAL=1.0
        idx = 5
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ni, nj = self.agent_pos[0] + di, self.agent_pos[1] + dj
                if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                    cell = self.grid[ni, nj]
                    obs[idx] = {
                        CellType.EMPTY: 0.0,
                        CellType.GOAL: 1.0,
                        CellType.OBSTACLE: -0.5,  # Obstacle (can be removed in future?)
                        CellType.FUEL: 0.5
                    }.get(cell, 0.0)
                else:
                    obs[idx] = -1.0  # Wall / out of bounds
                idx += 1
        
        return obs
    
    def _get_info(self) -> dict:
        return {
            'agent_position': tuple(self.agent_pos),
            'current_fuel': self.current_fuel,
            'fuel_collected': self.fuel_collected,
            'fuel_items_remaining': len(self.fuel_items),
            'steps_taken': self.steps_taken,
            'distance_to_goal': self._manhattan_distance(self.agent_pos, self.goal_pos),
            'total_reward': self.total_reward,
        }
    
    def _is_valid_position(self, pos) -> bool:
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
    
    def _manhattan_distance(self, p1, p2) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def render(self):
        if self.render_mode not in ["human", "ansi"]:
            return
        
        symbols = {CellType.EMPTY: '. ', CellType.GOAL: 'G ',
                  CellType.OBSTACLE: '# ', CellType.FUEL: 'F '}
        
        dist = self._manhattan_distance(self.agent_pos, self.goal_pos)
        
        lines = [f"Fuel: {self.current_fuel} | Distance: {dist} | Steps: {self.steps_taken}"]
        
        for i in range(self.grid_size):
            row = ""
            for j in range(self.grid_size):
                if [i, j] == self.agent_pos:
                    row += "A "
                else:
                    row += symbols[self.grid[i, j]]
            lines.append(row)
        
        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output


if __name__ == "__main__":
    print("Fuel Collection Environment")
    print("Observation: Goal direction + Fuel level + 5x5 view = 28 dims")
    print("A=Agent, G=Goal, F=Fuel, #=Obstacle\n")
    
    env = FuelCollectionEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    env.render()
    
    print(f"\nObservation shape: {obs.shape}")
    print(f"Goal direction: ({obs[0]:.2f}, {obs[1]:.2f})")
    print(f"Fuel level: {obs[2]:.2f}")
    print(f"5x5 view: {obs[3:].reshape(5,5)}")
