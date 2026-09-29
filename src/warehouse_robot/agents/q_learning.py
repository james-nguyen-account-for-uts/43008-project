from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random

import numpy as np

from warehouse_robot.environment import WarehouseEnv


@dataclass(slots=True)
class TrainingResult:
  episode_rewards: list[float]
  episode_steps: list[int]
  success_history: list[bool]

  @property
  def success_rate(self) -> float:
    if not self.success_history:
      return 0.0

    return float(np.mean(self.success_history))

  @property
  def average_reward(self) -> float:
    if not self.episode_rewards:
      return 0.0

    return float(np.mean(self.episode_rewards))

  @property
  def average_steps(self) -> float:
    if not self.episode_steps:
      return 0.0

    return float(np.mean(self.episode_steps))


class QLearningAgent:
  """Tabular Q-learning agent."""

  def __init__(
    self,
    action_count: int,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    minimum_epsilon: float = 0.05,
    seed: int = 42,
  ) -> None:
    self.action_count = action_count
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor

    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.minimum_epsilon = minimum_epsilon

    self.random_generator = random.Random(seed)
    np.random.seed(seed)

    self.q_table: defaultdict[
      tuple,
      np.ndarray,
    ] = defaultdict(lambda: np.zeros(
      self.action_count,
      dtype=np.float64,
    ))

  def choose_action(
    self,
    state: tuple,
    training: bool = True,
  ) -> int:
    """Select an action using an epsilon-greedy policy."""

    if (training and self.random_generator.random() < self.epsilon):
      return self.random_generator.randrange(self.action_count)

    action_values = self.q_table[state]

    best_actions = np.flatnonzero(action_values == np.max(action_values))

    return int(self.random_generator.choice(best_actions.tolist()))

  def update(
    self,
    state: tuple,
    action: int,
    reward: float,
    next_state: tuple,
    terminated: bool,
  ) -> None:
    """Apply the Q-learning update equation."""

    current_q_value = self.q_table[state][action]

    if terminated:
      target = reward
    else:
      target = (
        reward + self.discount_factor * np.max(self.q_table[next_state]))

    temporal_difference = target - current_q_value

    self.q_table[state][action] += (self.learning_rate * temporal_difference)

  def decay_exploration(self) -> None:
    """Reduce epsilon after each training episode."""

    self.epsilon = max(
      self.minimum_epsilon,
      self.epsilon * self.epsilon_decay,
    )

  def train(
    self,
    environment: WarehouseEnv,
    episodes: int = 5000,
    report_interval: int = 500,
  ) -> TrainingResult:
    """Train the agent in the warehouse environment."""

    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    success_history: list[bool] = []

    for episode in range(1, episodes + 1):
      environment.reset()
      state = environment.get_state_key()

      total_reward = 0.0
      terminated = False
      truncated = False
      info = {}

      while not terminated and not truncated:
        action = self.choose_action(
          state,
          training=True,
        )

        (
          _,
          reward,
          terminated,
          truncated,
          info,
        ) = environment.step(action)

        next_state = environment.get_state_key()

        self.update(
          state=state,
          action=action,
          reward=reward,
          next_state=next_state,
          terminated=terminated,
        )

        state = next_state
        total_reward += reward

      self.decay_exploration()

      episode_rewards.append(total_reward)
      episode_steps.append(info["steps"])
      success_history.append(info["success"])

      if episode % report_interval == 0:
        recent_success_rate = np.mean(success_history[-report_interval:])

        recent_average_reward = np.mean(episode_rewards[-report_interval:])

        print(
          f"Episode {episode:5d} | "
          f"Epsilon: {self.epsilon:.3f} | "
          f"Success: "
          f"{recent_success_rate * 100:6.2f}% | "
          f"Average reward: "
          f"{recent_average_reward:8.2f} | "
          f"Q-table states: {len(self.q_table)}")

    return TrainingResult(
      episode_rewards=episode_rewards,
      episode_steps=episode_steps,
      success_history=success_history,
    )

  def evaluate(
    self,
    environment: WarehouseEnv,
    episodes: int = 100,
  ) -> dict[str, float]:
    """Evaluate the learned greedy policy."""

    rewards: list[float] = []
    steps: list[int] = []
    invalid_actions: list[int] = []
    successes: list[bool] = []

    for _ in range(episodes):
      environment.reset()
      state = environment.get_state_key()

      total_reward = 0.0
      terminated = False
      truncated = False
      info = {}

      while not terminated and not truncated:
        action = self.choose_action(
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

      rewards.append(total_reward)
      steps.append(info["steps"])
      invalid_actions.append(info["invalid_actions"])
      successes.append(info["success"])

    return {
      "success_rate": float(np.mean(successes)),
      "average_reward": float(np.mean(rewards)),
      "average_steps": float(np.mean(steps)),
      "average_invalid_actions": float(np.mean(invalid_actions)),
    }
