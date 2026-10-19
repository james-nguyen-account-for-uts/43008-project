"""Watch a saved policy on any reachable generated map without retraining."""
import argparse
import secrets

from warehouse_robot.agents import NavigationQLearningAgent
from warehouse_robot.core.config import WarehouseConfig
from warehouse_robot.environment import WarehouseEnv
from warehouse_robot.visualisation import PygameVisualiser


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default="models/navigation_q.json")
  parser.add_argument("--seed", type=int, default=None, help="Omit for a new random map")
  parser.add_argument("--fps", type=int, default=8)
  args = parser.parse_args()
  if args.fps <= 0:
    parser.error("fps must be positive")
  try:
    agent, metadata = NavigationQLearningAgent.load(args.model)
  except FileNotFoundError:
    parser.error(f"Model not found: {args.model}. Run python train_q_learning.py first.")
  config = WarehouseConfig(**metadata["config"])
  seed = args.seed if args.seed is not None else secrets.randbits(32)
  environment = WarehouseEnv(config, layout_seed=seed)
  print(f"Map seed: {environment.layout_seed}. Click START; Space pauses; Escape closes.")
  print("Hybrid Q-learning with BFS guidance; delivery order is not guaranteed optimal.")
  visualiser = PygameVisualiser(environment, fps=args.fps)
  try:
    result = visualiser.run_trained_policy(agent)
  finally:
    visualiser.close()
  print(result)


if __name__ == "__main__":
  main()
