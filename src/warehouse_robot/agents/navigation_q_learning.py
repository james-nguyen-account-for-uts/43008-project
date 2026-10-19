"""Transferable Q-learning with explicit shortest-distance planning features.

BFS selects distance to the nearest box/unused finish. The Q-table learns
primitive actions on a small abstraction shared by maps, rather than cell IDs.
This is a hybrid baseline, not an end-to-end learned or globally optimal solver.
"""
import json
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np

from warehouse_robot.agents.q_learning import QLearningAgent


@lru_cache(maxsize=256)
def _distance_field(targets: tuple, obstacles: tuple, height: int, width: int) -> dict:
  """Reuse one BFS field for all robot positions in a delivery phase."""
  blocked = set(obstacles)
  distances = {target: 0 for target in targets}
  queue = deque(targets)
  while queue:
    r, c = queue.popleft()
    for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
      neighbour = (r + dr, c + dc)
      if (0 <= neighbour[0] < height and 0 <= neighbour[1] < width
          and neighbour not in blocked and neighbour not in distances):
        distances[neighbour] = distances[(r, c)] + 1
        queue.append(neighbour)
  return distances


@lru_cache(maxsize=20000)
def navigation_features(state: tuple) -> tuple[tuple, int]:
  row, col, carrying, boxes, used, obstacles, finishes, height, width = state
  targets = tuple(sorted(set(finishes) - set(used))) if carrying else boxes
  distances = _distance_field(targets, obstacles, height, width)
  distance = distances.get((row, col), 0)
  directions = []
  for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
    neighbour = (row + dr, col + dc)
    directions.append(distances[neighbour] - distance if neighbour in distances else 9)
  return (int(carrying), (row, col) in targets, *directions), distance


class NavigationQLearningAgent(QLearningAgent):
  """Learn reusable actions from BFS features across generated warehouses."""

  def choose_action(self, state: tuple, training: bool = True) -> int:
    features, _ = navigation_features(state)
    return super().choose_action(features, training=training)

  def update(self, state, action, reward, next_state, terminated):
    features, _ = navigation_features(state)
    next_features, _ = navigation_features(next_state)
    super().update(features, action, reward, next_features, terminated)

  def learning_reward(self, state, next_state, reward, terminated):
    # Potential-based shaping: never pay repeatedly for entering target cells.
    _, distance = navigation_features(state)
    _, next_distance = navigation_features(next_state)
    next_potential = 0 if terminated else -next_distance
    return reward + self.discount_factor * next_potential + distance

  def save(self, path, config: dict, training: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
      "format": "warehouse-navigation-q-v1",
      "config": config,
      "training": training,
      "discount_factor": self.discount_factor,
      "q_table": [[list(key), values.tolist()] for key, values in self.q_table.items()],
    }
    destination.write_text(json.dumps(payload, indent=2))

  @classmethod
  def load(cls, path):
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != "warehouse-navigation-q-v1":
      raise ValueError("Unsupported model format. Train with train_q_learning.py first.")
    agent = cls(action_count=6, epsilon=0.0, discount_factor=payload["discount_factor"])
    for key, values in payload["q_table"]:
      agent.q_table[tuple(key)] = np.asarray(values, dtype=np.float64)
    return agent, payload
