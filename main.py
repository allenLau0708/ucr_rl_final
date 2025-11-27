"""
Main Entry Point - Fuel Collection RL Project
CS258 Final Project: Learning to Survive in a Dual-Objective Environment

Authors: Zeli Liu, Hefeifei Jiang
"""

import argparse
from environment import FuelCollectionEnv


def demo_environment():
    """Demo the environment with manual/random play."""
    import time
    
    print("\n" + "="*60)
    print("  FUEL COLLECTION ENVIRONMENT DEMO")
    print("="*60)
    print()
    print("🤖 Agent   🏁 Goal   ⛽ Fuel   ██ Obstacle")
    print()
    print("CHALLENGE: You have LIMITED fuel (15 steps)")
    print("           Goal is ~22 steps away")
    print("           Collect ⛽ fuel to survive!")
    print()
    
    env = FuelCollectionEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    env.render()
    
    input("\nPress Enter to watch random agent...")
    
    done = False
    while not done:
        time.sleep(0.3)
        print("\033c", end="")  # Clear screen
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        print(f"Action: {['↑','↓','←','→'][action]}  Reward: {reward:+.1f}")
        env.render()
        
        if info.get('died'):
            print("\n💀 OUT OF FUEL!")
        elif info.get('reached_goal'):
            print(f"\n🎉 GOAL! Remaining fuel: {info['current_fuel']}")
    
    print(f"\nEfficiency: {info['efficiency_reward_total']:+.1f}")
    print(f"Collection: {info['collection_reward_total']:+.1f}")


def main():
    parser = argparse.ArgumentParser(
        description='Fuel Collection RL Project',
        epilog="""
Examples:
  python main.py --demo              # Demo environment
  python train.py --mode compare     # Train & compare agents
  python visualize.py                # Play manually (pygame)
  python visualize.py --model <name> # Watch trained agent
        """
    )
    parser.add_argument('--demo', action='store_true', help='Demo environment')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("CS258 Final Project: Dual-Objective RL")
    print("Fuel Collection Environment")
    print("Authors: Zeli Liu, Hefeifei Jiang")
    print("="*60)
    
    if args.demo:
        demo_environment()
    else:
        parser.print_help()
        print("\nQuick Start:")
        print("  1. python main.py --demo        # See the environment")
        print("  2. python train.py              # Train agents")
        print("  3. python visualize.py --list   # List trained models")
        print("  4. python visualize.py --model <name>  # Watch agent")


if __name__ == "__main__":
    main()
