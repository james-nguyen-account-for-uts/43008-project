from warehouse_robot.agents import QLearningAgent
from warehouse_robot.core.config import WarehouseConfig
from warehouse_robot.environment import WarehouseEnv

TRAINING_EPISODES = 5000
EVALUATION_EPISODES = 100
LAYOUT_SEED = 234567

# Output options
SHOW_DEMONSTRATION = True
SHOW_EACH_STEP = False


def main() -> None:
  config = WarehouseConfig(
    width=15,
    height=15,
    obstacle_count=10,
    box_count=3,
    finish_count=3,
    max_steps=250,
    seed=LAYOUT_SEED,
  )

  environment = WarehouseEnv(
    config=config,
    layout_seed=LAYOUT_SEED,
  )

  agent = QLearningAgent(
    action_count=environment.action_space.n,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.998,
    minimum_epsilon=0.05,
    seed=42,
  )

  print("=" * 60)
  print("Warehouse Robot Q-Learning")
  print("=" * 60)
  print(f"Layout seed       : {environment.layout_seed}")
  print(f"Grid size         : {config.width}x{config.height}")
  print(f"Obstacle blocks   : {config.obstacle_count}")
  print(f"Boxes             : {config.box_count}")
  print(f"Training episodes : {TRAINING_EPISODES}")
  print()

  print("Initial warehouse:")
  environment.reset()
  print(environment.render())
  print()

  training_result = agent.train(
    environment=environment,
    episodes=TRAINING_EPISODES,
    report_interval=500,
  )

  evaluation = agent.evaluate(
    environment=environment,
    episodes=EVALUATION_EPISODES,
  )

  print()
  print("=" * 60)
  print("Training summary")
  print("=" * 60)
  print(
    f"Overall training success rate : "
    f"{training_result.success_rate * 100:.2f}%")
  print(
    f"Average training reward       : "
    f"{training_result.average_reward:.2f}")
  print(
    f"Average training steps        : "
    f"{training_result.average_steps:.2f}")
  print(f"Q-table states                : "
        f"{len(agent.q_table)}")

  print()
  print("=" * 60)
  print("Evaluation results")
  print("=" * 60)
  print(
    f"Success rate            : "
    f"{evaluation['success_rate'] * 100:.2f}%")
  print(f"Average reward          : "
        f"{evaluation['average_reward']:.2f}")
  print(f"Average steps           : "
        f"{evaluation['average_steps']:.2f}")
  print(
    f"Average invalid actions : "
    f"{evaluation['average_invalid_actions']:.2f}")

  if SHOW_DEMONSTRATION:
    demonstrate_policy(
      environment=environment,
      agent=agent,
      show_each_step=SHOW_EACH_STEP,
    )


def demonstrate_policy(
  environment: WarehouseEnv,
  agent: QLearningAgent,
  show_each_step: bool = False,
) -> None:
  """
  Run one episode using the trained greedy policy.

  If show_each_step is False, only the initial state,
  final state, and demonstration summary are printed.
  """

  environment.reset()
  state = environment.get_state_key()

  terminated = False
  truncated = False
  total_reward = 0.0

  print()
  print("=" * 60)
  print("Trained policy demonstration")
  print("=" * 60)

  print("Initial state:")
  print(environment.render())

  while not terminated and not truncated:
    action = agent.choose_action(
      state,
      training=False,
    )

    (
      _,
      reward,
      terminated,
      truncated,
      info,
    ) = environment.step(action)

    state = environment.get_state_key()
    total_reward += reward

    if show_each_step:
      print()
      print(
        f"Step: {info['steps']} | "
        f"Action: {action} | "
        f"Reward: {reward:.1f}")
      print(environment.render())

  print()
  print("Final state:")
  print(environment.render())

  print()
  print("=" * 60)
  print("Demonstration summary")
  print("=" * 60)
  print(f"Success         : {info['success']}")
  print(f"Total reward    : {total_reward:.2f}")
  print(f"Total steps     : {info['steps']}")
  print(f"Boxes delivered : {info['delivered']}")
  print(f"Invalid actions : {info['invalid_actions']}")

  if terminated:
    print("Result          : All boxes delivered successfully")
  elif truncated:
    print("Result          : Episode reached the step limit")


if __name__ == "__main__":
  main()
