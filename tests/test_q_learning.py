import numpy as np
import pytest

from warehouse_robot.agents import (
  QLearningAgent,
  TrainingResult,
)
from warehouse_robot.core.config import WarehouseConfig
from warehouse_robot.environment import WarehouseEnv


@pytest.fixture
def agent() -> QLearningAgent:
  return QLearningAgent(
    action_count=6,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=0.0,
    epsilon_decay=0.9,
    minimum_epsilon=0.05,
    seed=42,
  )


def test_new_state_has_zero_q_values(agent: QLearningAgent, ) -> None:
  state = ("sample", )

  values = agent.q_table[state]

  np.testing.assert_array_equal(
    values,
    np.zeros(6),
  )


def test_agent_selects_best_greedy_action(agent: QLearningAgent, ) -> None:
  state = ("sample", )

  agent.q_table[state] = np.array([
    0.0,
    2.0,
    1.0,
    10.0,
    -1.0,
    4.0,
  ])

  action = agent.choose_action(
    state,
    training=False,
  )

  assert action == 3


def test_agent_uses_random_action_during_exploration() -> None:
  agent = QLearningAgent(
    action_count=6,
    epsilon=1.0,
    seed=42,
  )

  state = ("sample", )

  selected_actions = {
    agent.choose_action(
      state,
      training=True,
    )
    for _ in range(100)
  }

  assert len(selected_actions) > 1

  assert all(0 <= action < 6 for action in selected_actions)


def test_terminal_q_learning_update(agent: QLearningAgent, ) -> None:
  state = ("current", )
  next_state = ("next", )

  agent.update(
    state=state,
    action=2,
    reward=10.0,
    next_state=next_state,
    terminated=True,
  )

  # Q = 0 + 0.1 * (10 - 0)
  assert agent.q_table[state][2] == pytest.approx(1.0)


def test_non_terminal_q_learning_update(agent: QLearningAgent, ) -> None:
  state = ("current", )
  next_state = ("next", )

  agent.q_table[next_state] = np.array([
    0.0,
    4.0,
    2.0,
    0.0,
    0.0,
    0.0,
  ])

  agent.update(
    state=state,
    action=1,
    reward=2.0,
    next_state=next_state,
    terminated=False,
  )

  # Target = 2 + 0.95 * 4 = 5.8
  # Updated Q = 0 + 0.1 * 5.8 = 0.58
  assert agent.q_table[state][1] == pytest.approx(0.58)


def test_epsilon_decays(agent: QLearningAgent, ) -> None:
  agent.epsilon = 1.0
  agent.decay_exploration()

  assert agent.epsilon == pytest.approx(0.9)


def test_epsilon_does_not_fall_below_minimum(agent: QLearningAgent, ) -> None:
  agent.epsilon = 0.051

  for _ in range(100):
    agent.decay_exploration()

  assert agent.epsilon == pytest.approx(0.05)


def test_training_result_properties() -> None:
  result = TrainingResult(
    episode_rewards=[10.0, 20.0, 30.0],
    episode_steps=[30, 20, 10],
    success_history=[False, True, True],
  )

  assert result.average_reward == pytest.approx(20.0)
  assert result.average_steps == pytest.approx(20.0)
  assert result.success_rate == pytest.approx(2 / 3)


def test_empty_training_result_properties() -> None:
  result = TrainingResult(
    episode_rewards=[],
    episode_steps=[],
    success_history=[],
  )

  assert result.average_reward == 0.0
  assert result.average_steps == 0.0
  assert result.success_rate == 0.0


def test_short_training_run_completes() -> None:
  config = WarehouseConfig(
    width=6,
    height=6,
    obstacle_count=1,
    box_count=1,
    finish_count=1,
    max_steps=10,
    seed=42,
  )

  environment = WarehouseEnv(
    config=config,
    layout_seed=42,
  )

  agent = QLearningAgent(
    action_count=environment.action_space.n,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.9,
    minimum_epsilon=0.05,
    seed=42,
  )

  result = agent.train(
    environment=environment,
    episodes=5,
    report_interval=10,
  )

  assert len(result.episode_rewards) == 5
  assert len(result.episode_steps) == 5
  assert len(result.success_history) == 5
  assert len(agent.q_table) > 0


def test_evaluation_returns_expected_metrics() -> None:
  config = WarehouseConfig(
    width=6,
    height=6,
    obstacle_count=1,
    box_count=1,
    finish_count=1,
    max_steps=5,
    seed=42,
  )

  environment = WarehouseEnv(
    config=config,
    layout_seed=42,
  )

  agent = QLearningAgent(
    action_count=environment.action_space.n,
    epsilon=0.0,
    seed=42,
  )

  results = agent.evaluate(
    environment=environment,
    episodes=2,
  )

  assert set(results) == {
    "success_rate",
    "average_reward",
    "average_steps",
    "average_invalid_actions",
  }

  assert 0.0 <= results["success_rate"] <= 1.0
  assert results["average_steps"] <= 5
  assert results["average_invalid_actions"] >= 0
