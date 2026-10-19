"""Train a transferable hybrid Q-learning policy; evaluate on unseen maps."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from warehouse_robot.agents import NavigationQLearningAgent
from warehouse_robot.core.config import WarehouseConfig
from warehouse_robot.environment import WarehouseEnv
from warehouse_robot.evaluation_report import write_evaluation_report


def parse_arguments():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--episodes", type=int, default=3000)
  parser.add_argument("--eval-maps", type=int, default=100)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--width", type=int, default=15)
  parser.add_argument("--height", type=int, default=15)
  parser.add_argument("--obstacles", type=int, default=10)
  parser.add_argument("--boxes", type=int, default=3)
  parser.add_argument("--max-steps", type=int, default=500)
  parser.add_argument("--model", default="models/navigation_q.json")
  args = parser.parse_args()
  if args.episodes <= 0 or args.eval_maps <= 0 or args.boxes <= 0:
    parser.error("episodes, eval-maps, and boxes must be positive")
  return args


def main():
  args = parse_arguments()
  config = WarehouseConfig(width=args.width, height=args.height,
    obstacle_count=args.obstacles, box_count=args.boxes, finish_count=args.boxes,
    max_steps=args.max_steps, seed=args.seed)
  environment = WarehouseEnv(config, layout_seed=args.seed, randomize_layout=True)
  # Seed the random-map stream once. Subsequent resets draw a new layout seed.
  environment.reset(seed=args.seed)
  agent = NavigationQLearningAgent(action_count=6, epsilon_decay=0.998, seed=args.seed)
  print("Training hybrid Q-learning on a new generated map each episode.", flush=True)
  result = agent.train(environment, episodes=args.episodes,
                       report_interval=max(1, min(100, args.episodes)))

  # Training seeds come from [0, 2**31); evaluation uses a disjoint range.
  evaluation_seeds = [2**32 + i * 1000 for i in range(args.eval_maps)]
  records = []
  for seed in evaluation_seeds:
    evaluation_env = WarehouseEnv(config, layout_seed=seed)
    metrics = agent.evaluate(evaluation_env, episodes=1)
    records.append({"layout_seed": evaluation_env.layout_seed, **metrics})
  summary = {key: sum(row[key] for row in records) / len(records)
             for key in ("success_rate", "average_reward", "average_steps", "average_invalid_actions")}
  metadata = {"episodes": args.episodes, "seed": args.seed,
              "training_success_rate": result.success_rate, "evaluation": summary,
              "training_history": {"rewards": result.episode_rewards,
                                   "steps": result.episode_steps,
                                   "success": result.success_history},
              "evaluation_maps": records,
              "method": "Q-learning on BFS nearest-target features; no global optimality guarantee"}
  agent.save(args.model, asdict(config), metadata)
  report = Path(args.model).with_suffix(".evaluation.json")
  report.write_text(json.dumps(metadata, indent=2))
  dashboard = write_evaluation_report(metadata, report.with_suffix(".html"))
  print(f"Saved dashboard: {dashboard}")
  print("Unseen-map evaluation:", json.dumps(summary, indent=2))
  print(f"Saved model: {args.model}")
  print(f"Saved evaluation: {report}")
  print(f"Watch: python visualise_policy.py --model {args.model}")


if __name__ == "__main__":
  main()
