import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from dataclasses import asdict

import numpy as np

from warehouse_robot.agents import NavigationQLearningAgent
from warehouse_robot.agents.navigation_q_learning import navigation_features
from warehouse_robot.core.config import WarehouseConfig
from warehouse_robot.core.types import Action
from warehouse_robot.environment import WarehouseEnv


def config():
  return WarehouseConfig(width=8, height=8, obstacle_count=2, box_count=1, finish_count=1)


def test_random_map_stream_is_varied_and_reproducible():
  first = WarehouseEnv(config(), randomize_layout=True)
  second = WarehouseEnv(config(), randomize_layout=True)
  first.reset(seed=123)
  second.reset(seed=123)
  states = []
  for _ in range(4):
    observation, _ = first.reset()
    second.reset()
    assert first.get_state_key() == second.get_state_key()
    assert first.observation_space.contains(observation)
    assert first._all_targets_reachable(first.initial_state)
    states.append(first.get_state_key())
  assert len(set(states)) == 4


def test_seeded_reset_generates_requested_layout():
  env = WarehouseEnv(config())
  env.reset(seed=456)
  expected = WarehouseEnv(config(), layout_seed=456)
  assert env.get_state_key() == expected.get_state_key()


def test_layout_is_part_of_state_and_observation():
  env = WarehouseEnv(config())
  observation, _ = env.reset()
  assert observation["obstacles"].sum() == len(env.initial_state.obstacle_positions)
  assert len(env.get_state_key()) == 9


def test_navigation_features_transfer_between_translated_maps():
  # Same useful eastward move in different absolute positions.
  one = (1, 1, 0, ((1, 3),), (), (), ((3, 3),), 5, 5)
  two = (2, 2, 0, ((2, 4),), (), (), ((4, 4),), 6, 6)
  assert navigation_features(one)[0] == navigation_features(two)[0]
  agent = NavigationQLearningAgent(action_count=6, epsilon=0)
  agent.update(one, Action.RIGHT, 10, one, True)
  assert agent.choose_action(two, training=False) == Action.RIGHT


def test_movement_target_does_not_pay_repeatable_bonus():
  env = WarehouseEnv(config())
  env.reset()
  target = next(iter(env.boxes))
  # Find an adjacent valid position and move onto the box.
  for action in (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT):
    opposite = Action((int(action) + 2) % 4)
    start = target.moved(opposite)
    if env.simulation.is_inside(start) and start not in env.initial_state.obstacle_positions:
      env.robot_position = start
      assert env.step(action)[1] == -1
      return
  raise AssertionError("No accessible neighbour")


def test_model_round_trip(tmp_path):
  env = WarehouseEnv(config())
  state = env.get_state_key()
  agent = NavigationQLearningAgent(action_count=6, epsilon=0)
  agent.update(state, 2, 10, state, True)
  path = tmp_path / "agent.json"
  agent.save(path, asdict(config()), {"episodes": 1})
  restored, metadata = NavigationQLearningAgent.load(path)
  assert restored.choose_action(state, training=False) == 2
  assert metadata["config"] == asdict(config())
  np.testing.assert_array_equal(list(agent.q_table.values()), list(restored.q_table.values()))


def test_gui_waits_before_first_action(monkeypatch):
  from warehouse_robot.visualisation import PygameVisualiser
  env = WarehouseEnv(config())
  visualiser = PygameVisualiser(env, fps=1000)
  agent = NavigationQLearningAgent(action_count=6)
  def stop_at_start():
    assert env.current_step == 0
    visualiser.running = False
  monkeypatch.setattr(visualiser, "_wait_for_start", stop_at_start)
  result = visualiser.run_trained_policy(agent)
  assert result["steps"] == 0
