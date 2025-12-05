"""
Pygame Visualization for Fuel Collection Environment.
CS258 Final Project

Features:
- White background with clear colors
- Random environment on each reset
- Toggle between Agent View (5x5) and Global View
"""

import os
from pathlib import Path
import torch
import numpy as np

try:
    import pygame
except ImportError:
    os.system("pip install pygame")
    import pygame

from environment import FuelCollectionEnv, CellType
from agent import ActorCriticNetwork
from agent_multi_critic import MultiCriticNetwork
from config import PPOConfig


# Light theme colors
COLORS = {
    'bg': (245, 245, 245),
    'grid_line': (200, 200, 200),
    'empty': (255, 255, 255),
    'agent': (41, 128, 185),
    'goal': (39, 174, 96),
    'fuel': (243, 156, 18),
    'fuel_collected': (200, 230, 200),  # Light green for collected fuel
    'obstacle': (52, 73, 94),
    'path': (155, 89, 182),
    'visible': (230, 240, 255),
    'visible_border': (100, 149, 237),
    'fog': (180, 180, 180),  # Fog of war for agent view
    'text': (44, 62, 80),
    'panel': (255, 255, 255),
    'bar_ok': (39, 174, 96),
    'bar_low': (231, 76, 60),
}


class Visualizer:
    def __init__(self, model_path: str = None, cell_size: int = 45, fps: int = 5):
        self.cell_size = cell_size
        self.fps = fps
        
        self.env = FuelCollectionEnv()
        self.grid_size = self.env.grid_size
        
        # Load model
        self.network = None
        self.manual_mode = True
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
            self.manual_mode = False
        
        # Window
        self.grid_width = self.grid_size * cell_size
        self.panel_width = 220
        self.window_width = self.grid_width + self.panel_width
        self.window_height = self.grid_size * cell_size
        
        pygame.init()
        pygame.display.set_caption("Fuel Collection RL")
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.font_large = pygame.font.Font(None, 28)
        self.font_icon = pygame.font.Font(None, 40)  # Larger font for icons
        self.asset_dir = Path(__file__).resolve().parent / "assets" / "icons"
        
        # Emoji font for icons (use system font that supports emoji)
        # Use SysFont lookup with common emoji-capable fonts
        self.emoji_font = None
        self.emoji_ready = False
        candidate_fonts = [
            "applecoloremoji",
            "arialunicodems",
            "segoeuiemoji",
            "notocoloremoji",
            "symbola",
            "menlo",
            None,
        ]
        for name in candidate_fonts:
            try:
                font = pygame.font.SysFont(name, 35)
                if font.render("⛽", True, (0, 0, 0)).get_width() > 5:
                    self.emoji_font = font
                    self.emoji_ready = True
                    print(f"Success loading font: {name}")
                    break
            except Exception:
                continue
        
        # Fallback to default font if no emoji-capable font found
        if not self.emoji_font:
            self.emoji_font = pygame.font.Font(None, 36)
        
        # Icon sets (emoji or ASCII fallback)
        self.ICON_SET_EMOJI = {
            'agent': '🤖',
            'goal': '🏁',
            'fuel': '⛽',
            'fuel_collected': '✅',
            'obstacle': '🧱',
            'unknown': '❓',
        }
        self.ICON_SET_ASCII = {
            'agent': 'A',       # Agent - simple letter
            'goal': 'G',        # Goal - simple letter
            'fuel': 'F',        # Fuel - simple letter
            'fuel_collected': '✓',  # Checkmark (Unicode)
            'obstacle': '█',    # Full block (Unicode, more visible)
            'unknown': '?',     # Question mark
        }
        self.use_emoji_icons = self.emoji_ready
        
        # Icons using Unicode symbols (emoji when supported)
        self.ICONS = self.ICON_SET_EMOJI if self.use_emoji_icons else self.ICON_SET_ASCII
        self.emoji_icon_values = set(self.ICON_SET_EMOJI.values())

        # Optional PNG/SVG (converted) icon images
        self.icon_image_files = {
            'agent': 'agent.png',
            'goal': 'goal.png',
            'fuel': 'fuel.png',
            'fuel_collected': 'fuel_collected.png',
            'obstacle': 'obstacle.png',
            'unknown': 'unknown.png',
        }
        self.icon_images = {}
        self.icon_images_small = {}
        self.use_image_icons = False
        self._load_image_icons()
        
        # Colors for icons
        self.ICON_COLORS = {
            'agent': (41, 128, 185),      # Blue
            'goal': (39, 174, 96),        # Green
            'fuel': (243, 156, 18),       # Orange
            'fuel_collected': (39, 174, 96),  # Green
            'obstacle': (52, 73, 94),     # Dark gray
            'unknown': (120, 120, 120),   # Gray
        }
        
        self.trajectory = []
        self.action_keys = {
            pygame.K_UP: 0, pygame.K_w: 0,
            pygame.K_DOWN: 1, pygame.K_s: 1,
            pygame.K_LEFT: 2, pygame.K_a: 2,
            pygame.K_RIGHT: 3, pygame.K_d: 3,
        }
        
        self.reset_count = 0
        self.global_view = True  # Toggle between global and agent view
        
    def _load_model(self, path: str):
        print(f"Loading: {path}")
        data = torch.load(path, map_location='cpu', weights_only=False)
        config = data.get('config', PPOConfig())
        
        # Check if this is a multi-critic model by examining state_dict keys
        state_dict = data['network']
        is_multi_critic = any('critics.' in key for key in state_dict.keys())
        
        if is_multi_critic:
            # Multi-critic model
            reward_terms = data.get('reward_terms', ['goal', 'fuel', 'survival'])
            self.network = MultiCriticNetwork(30, 4, config.hidden_sizes, reward_terms)
            print(f"  Detected Multi-Critic model with {len(reward_terms)} critics: {', '.join(reward_terms)}")
        else:
            # Standard single-critic model
            self.network = ActorCriticNetwork(30, 4, config.hidden_sizes)
            print(f"  Detected standard Actor-Critic model")
        
        self.network.load_state_dict(state_dict)
        self.network.eval()
    
    def _is_visible(self, i: int, j: int) -> bool:
        """Check if cell is within agent's 5x5 view."""
        agent_i, agent_j = self.env.agent_pos
        return abs(i - agent_i) <= 2 and abs(j - agent_j) <= 2

    def _load_image_icons(self):
        """Load PNG icon files if the user provides them."""
        if not self.asset_dir.exists():
            return
        icon_size_main = int(self.cell_size * 0.75)
        icon_size_small = 16  # for mini view
        loaded_any = False
        for key, filename in self.icon_image_files.items():
            path = self.asset_dir / filename
            if not path.exists():
                continue
            try:
                image = pygame.image.load(str(path)).convert_alpha()
                self.icon_images[key] = pygame.transform.smoothscale(image, (icon_size_main, icon_size_main))
                self.icon_images_small[key] = pygame.transform.smoothscale(image, (icon_size_small, icon_size_small))
                loaded_any = True
            except Exception as exc:
                print(f"Failed to load icon '{path}': {exc}")
        self.use_image_icons = loaded_any
    
    def _draw_grid(self):
        """Draw the grid with cells and emoji icons."""
        agent_i, agent_j = self.env.agent_pos
        
        # Track collected fuel positions
        current_fuel_set = set(tuple(f) for f in self.env.fuel_items)
        if hasattr(self, 'initial_fuel_positions'):
            self.collected_fuel_positions = self.initial_fuel_positions - current_fuel_set
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x, y = j * self.cell_size, i * self.cell_size
                cell = self.env.grid[i, j]
                pos = (i, j)
                
                # Check visibility in agent view mode
                is_visible = self._is_visible(i, j)
                
                # Check if this is a collected fuel position
                is_collected_fuel = hasattr(self, 'collected_fuel_positions') and pos in self.collected_fuel_positions
                
                # Determine cell color
                if cell == CellType.OBSTACLE:
                    color = COLORS['obstacle']
                elif cell == CellType.GOAL:
                    color = COLORS['goal']
                elif cell == CellType.FUEL:
                    color = COLORS['fuel']
                elif is_collected_fuel:
                    color = COLORS['fuel_collected']
                else:
                    color = COLORS['empty']

                # When using image icons, keep background neutral so PNGs have no color halo
                if self.use_image_icons and (cell in (CellType.GOAL, CellType.FUEL, CellType.OBSTACLE) or is_collected_fuel):
                    color = COLORS['empty']
                
                # Apply fog of war in agent view mode
                if not self.global_view and not is_visible:
                    color = COLORS['fog']
                
                # Draw cell background
                pygame.draw.rect(self.screen, color, (x+1, y+1, self.cell_size-2, self.cell_size-2))
                
                # Draw grid lines
                pygame.draw.rect(self.screen, COLORS['grid_line'], 
                               (x, y, self.cell_size, self.cell_size), 1)
                
                # Add icons (only if visible or global view)
                cx, cy = x + self.cell_size//2, y + self.cell_size//2
                if self.global_view or is_visible:
                    if cell == CellType.GOAL:
                        self._draw_icon(cx, cy, 'goal')
                    elif cell == CellType.FUEL:
                        self._draw_icon(cx, cy, 'fuel')
                    elif cell == CellType.OBSTACLE:
                        self._draw_icon(cx, cy, 'obstacle')
                    elif is_collected_fuel:
                        self._draw_icon(cx, cy, 'fuel_collected')
                elif not self.global_view:
                    # Draw ? for unknown cells
                    self._draw_icon(cx, cy, 'unknown')
    
    def _draw_icon(self, cx, cy, icon_key):
        """Draw an icon at the center position."""
        icon_char = self.ICONS.get(icon_key, '?')
        icon_color = self.ICON_COLORS.get(icon_key, (0, 0, 0))

        # Prefer image icons when provided
        if self.use_image_icons:
            image = self.icon_images.get(icon_key)
            if image is not None:
                rect = image.get_rect(center=(cx, cy))
                self.screen.blit(image, rect)
                return
        
        # Try emoji font for emoji characters first
        if self.use_emoji_icons and icon_char in self.emoji_icon_values and self.emoji_font:
            try:
                text = self.emoji_font.render(icon_char, True, icon_color)
                if text.get_width() > 5:  # Check if actually rendered
                    text_rect = text.get_rect(center=(cx, cy))
                    self.screen.blit(text, text_rect)
                    return
            except:
                # If emoji render fails mid-run, disable and fall back
                self.use_emoji_icons = False
                self.ICONS = self.ICON_SET_ASCII
        
        # Use regular font with colored Unicode symbols
        text = self.font_icon.render(icon_char, True, icon_color)
        text_rect = text.get_rect(center=(cx, cy))
        self.screen.blit(text, text_rect)
        
        # Draw visible area border in agent view mode
        if not self.global_view:
            agent_i, agent_j = self.env.agent_pos
            min_i = max(0, agent_i - 2)
            max_i = min(self.grid_size - 1, agent_i + 2)
            min_j = max(0, agent_j - 2)
            max_j = min(self.grid_size - 1, agent_j + 2)
            
            x = min_j * self.cell_size
            y = min_i * self.cell_size
            w = (max_j - min_j + 1) * self.cell_size
            h = (max_i - min_i + 1) * self.cell_size
            
            pygame.draw.rect(self.screen, COLORS['visible_border'], (x, y, w, h), 3)
    
    def _draw_trajectory(self):
        """Draw agent's path."""
        if len(self.trajectory) > 1:
            # In agent view, only show visible parts of trajectory
            if self.global_view:
                points = [(p[1] * self.cell_size + self.cell_size//2,
                          p[0] * self.cell_size + self.cell_size//2) for p in self.trajectory]
                pygame.draw.lines(self.screen, COLORS['path'], False, points, 2)
            else:
                # Only draw visible trajectory segments
                visible_points = []
                for p in self.trajectory:
                    if self._is_visible(p[0], p[1]):
                        visible_points.append((p[1] * self.cell_size + self.cell_size//2,
                                              p[0] * self.cell_size + self.cell_size//2))
                if len(visible_points) > 1:
                    pygame.draw.lines(self.screen, COLORS['path'], False, visible_points, 2)
    
    def _draw_agent(self):
        """Draw the agent using icon."""
        pos = self.env.agent_pos
        cx = pos[1] * self.cell_size + self.cell_size // 2
        cy = pos[0] * self.cell_size + self.cell_size // 2
        
        # Draw agent icon
        self._draw_icon(cx, cy, 'agent')
    
    def _draw_panel(self, info: dict):
        """Draw info panel."""
        px = self.grid_width
        
        # Panel background
        pygame.draw.rect(self.screen, COLORS['panel'], (px, 0, self.panel_width, self.window_height))
        pygame.draw.line(self.screen, COLORS['grid_line'], (px, 0), (px, self.window_height), 2)
        
        y = 15
        
        # Mode indicator
        mode = "AI" if not self.manual_mode else "Manual"
        text = self.font_large.render(f"Mode: {mode}", True, COLORS['text'])
        self.screen.blit(text, (px + 10, y))
        y += 30
        
        # View mode toggle indicator
        view_mode = "Global View" if self.global_view else "Agent View (5×5)"
        view_color = COLORS['goal'] if self.global_view else COLORS['visible_border']
        text = self.font.render(view_mode, True, view_color)
        self.screen.blit(text, (px + 10, y))
        y += 25
        
        # Map number
        text = self.font.render(f"Map: #{self.reset_count}", True, COLORS['text'])
        self.screen.blit(text, (px + 10, y))
        y += 30
        
        # Fuel bar
        fuel = info.get('current_fuel', 0)
        dist = info.get('distance_to_goal', 0)
        
        text = self.font.render(f"Fuel: {fuel}", True, COLORS['text'])
        self.screen.blit(text, (px + 10, y))
        y += 22
        
        bar_w = 180
        pygame.draw.rect(self.screen, (220, 220, 220), (px+10, y, bar_w, 16), border_radius=3)
        fill = min(1.0, fuel / 30) * bar_w
        color = COLORS['bar_ok'] if fuel >= dist else COLORS['bar_low']
        if fill > 0:
            pygame.draw.rect(self.screen, color, (px+10, y, int(fill), 16), border_radius=3)
        y += 30
        
        # Stats
        stats = [
            f"Distance: {dist}",
            f"Steps: {info.get('steps_taken', 0)}",
            f"Collected: {info.get('fuel_collected', 0)}",
            f"Reward: {info.get('total_reward', 0):.0f}",
        ]
        for s in stats:
            text = self.font.render(s, True, COLORS['text'])
            self.screen.blit(text, (px + 10, y))
            y += 22
        
        # Agent's local view (mini map)
        y += 10
        pygame.draw.line(self.screen, COLORS['grid_line'], (px+10, y), (px+190, y))
        y += 10
        text = self.font.render("Agent's View:", True, COLORS['visible_border'])
        self.screen.blit(text, (px + 10, y))
        y += 22
        
        # Draw mini view
        agent_i, agent_j = self.env.agent_pos
        mini_size = 20
        start_x = px + 15
        start_y = y
        
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ni, nj = agent_i + di, agent_j + dj
                mx = start_x + (dj + 2) * mini_size
                my = start_y + (di + 2) * mini_size
                
                icon_key = None
                if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                    cell = self.env.grid[ni, nj]
                    pos = (ni, nj)
                    if di == 0 and dj == 0:
                        color = COLORS['empty'] if self.use_image_icons else COLORS['agent']
                        icon_key = 'agent'
                    elif cell == CellType.OBSTACLE:
                        color = COLORS['empty'] if self.use_image_icons else COLORS['obstacle']
                        icon_key = 'obstacle'
                    elif cell == CellType.GOAL:
                        color = COLORS['empty'] if self.use_image_icons else COLORS['goal']
                        icon_key = 'goal'
                    elif cell == CellType.FUEL:
                        color = COLORS['empty'] if self.use_image_icons else COLORS['fuel']
                        icon_key = 'fuel'
                    elif hasattr(self, 'collected_fuel_positions') and pos in self.collected_fuel_positions:
                        color = COLORS['empty']
                        icon_key = 'fuel_collected'
                    else:
                        color = COLORS['empty']
                else:
                    color = (100, 100, 100)  # Out of bounds (wall)
                
                pygame.draw.rect(self.screen, color, (mx, my, mini_size-1, mini_size-1))
                if self.use_image_icons and icon_key:
                    image = self.icon_images_small.get(icon_key)
                    if image:
                        rect = image.get_rect(center=(mx + mini_size//2, my + mini_size//2))
                        self.screen.blit(image, rect)
                pygame.draw.rect(self.screen, COLORS['grid_line'], (mx, my, mini_size, mini_size), 1)
        
        # Controls
        y = self.window_height - 110
        pygame.draw.line(self.screen, COLORS['grid_line'], (px+10, y), (px+190, y))
        y += 8
        controls = [
            "V: Toggle View",
            "R: New Map",
            # "↑↓←→: Move",
            "U D L R: Move",
            "SPACE: Pause",
            "Q: Quit"
        ]
        for c in controls:
            text = self.font.render(c, True, (120, 120, 120))
            self.screen.blit(text, (px + 10, y))
            y += 18
    
    def run(self, seed: int = None):
        """Main loop."""
        if seed is None:
            seed = np.random.randint(0, 100000)
        
        obs, info = self.env.reset(seed=seed)
        self.reset_count = 1
        self.trajectory = [list(self.env.agent_pos)]
        self.initial_fuel_positions = set(tuple(f) for f in self.env.fuel_items)
        self.collected_fuel_positions = set()
        
        done = False
        paused = False
        pending_action = None
        
        print("\nControls:")
        print("  V: Toggle Global/Agent View")
        print("  R: New Random Map")
        # print("  ↑↓←→: Move")
        print("  U D L R: Move")
        print("  SPACE: Pause")
        print("  Q: Quit")
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                        pygame.quit()
                        return
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_v:
                        # Toggle view mode
                        self.global_view = not self.global_view
                    elif event.key == pygame.K_r:
                        seed = np.random.randint(0, 100000)
                        obs, info = self.env.reset(seed=seed)
                        self.reset_count += 1
                        self.trajectory = [list(self.env.agent_pos)]
                        done = False
                    elif event.key in self.action_keys:
                        pending_action = self.action_keys[event.key]
                        self.manual_mode = True
            
            self.screen.fill(COLORS['bg'])
            
            if not paused and not done:
                if self.manual_mode:
                    if pending_action is not None:
                        obs, _, term, trunc, info = self.env.step(pending_action)
                        done = term or trunc
                        self.trajectory.append(list(self.env.agent_pos))
                        pending_action = None
                elif self.network:
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        # Handle both standard and multi-critic networks
                        if isinstance(self.network, MultiCriticNetwork):
                            logits, _ = self.network.forward(obs_tensor)
                        else:
                            logits, _ = self.network.forward(obs_tensor)
                        action = torch.argmax(torch.softmax(logits, dim=-1)).item()
                    obs, _, term, trunc, info = self.env.step(action)
                    done = term or trunc
                    self.trajectory.append(list(self.env.agent_pos))
            
            self._draw_grid()
            self._draw_trajectory()
            self._draw_agent()
            self._draw_panel(info)
            
            if done:
                msg = "GOAL!" if info.get('reached_goal') else "DEAD!"
                color = COLORS['goal'] if info.get('reached_goal') else COLORS['bar_low']
                text = self.font_large.render(msg, True, color)
                rect = text.get_rect(center=(self.grid_width//2, self.window_height//2))
                bg_rect = rect.inflate(30, 15)
                pygame.draw.rect(self.screen, (255, 255, 255), bg_rect, border_radius=8)
                pygame.draw.rect(self.screen, color, bg_rect, 2, border_radius=8)
                self.screen.blit(text, rect)
            
            if paused:
                text = self.font_large.render("PAUSED", True, COLORS['fuel'])
                self.screen.blit(text, text.get_rect(center=(self.grid_width//2, 25)))
            
            pygame.display.flip()
            self.clock.tick(self.fps if not self.manual_mode else 30)


def list_models():
    models = {}
    if os.path.exists("models"):
        for f in os.listdir("models"):
            if f.endswith(".pt"):
                models[f.replace(".pt", "")] = os.path.join("models", f)
    return models


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model path')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--agent-view', action='store_true', help='Start in agent view mode')
    args = parser.parse_args()
    
    models = list_models()
    
    if args.list:
        print("Models:", list(models.keys()) if models else "(none)")
        return
    
    model_path = None
    if args.model:
        model_path = models.get(args.model, args.model)
        if not os.path.exists(model_path):
            print(f"Not found: {args.model}")
            return
    
    viz = Visualizer(model_path, fps=args.fps)
    if args.agent_view:
        viz.global_view = False
    viz.run()


if __name__ == "__main__":
    main()
