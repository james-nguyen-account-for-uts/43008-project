from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from warehouse_robot.core.config import WarehouseConfig
from warehouse_robot.core.types import Action, Position, Tile
from warehouse_robot.simulation.warehouse import (
  WarehouseSimulation,
  WarehouseState,
)


class WarehouseEnv(gym.Env):
  """
    Gymnasium environment for warehouse robot navigation.

    The robot must:

    1. Navigate to a box.
    2. Use PICK_UP while standing on the box.
    3. Navigate to an unused finish zone.
    4. Use DROP_OFF while standing on the finish zone.
    """

  metadata = {
    "render_modes": ["ansi"],
    "render_fps": 4,
  }

  def __init__(
    self,
    config: WarehouseConfig,
    layout_seed: int = 42,
    render_mode: str | None = None,
  ) -> None:
    super().__init__()

    self.config = config
    self.layout_seed = layout_seed
    self.render_mode = render_mode

    self.simulation = WarehouseSimulation(config)

    # Six actions from the Action enum.
    self.action_space = spaces.Discrete(len(Action))

    total_cells = config.width * config.height

    self.observation_space = spaces.Dict(
      {
        "robot": spaces.MultiDiscrete([
          config.height,
          config.width,
        ]),
        "boxes": spaces.MultiBinary(total_cells),
        "finish_zones": spaces.MultiBinary(total_cells),
        "carrying": spaces.Discrete(2),
        "delivered": spaces.Discrete(config.box_count + 1),
      })

    self.initial_state = self._generate_reachable_layout(layout_seed)

    self.robot_position = self.initial_state.robot
    self.boxes = set(self.initial_state.boxes)
    self.finish_zones = set(self.initial_state.finish_zones)
    self.used_finish_zones: set[Position] = set()

    self.carrying_box = False
    self.delivered_count = 0
    self.current_step = 0
    self.invalid_action_count = 0

  def _generate_reachable_layout(
    self,
    starting_seed: int,
    maximum_attempts: int = 100,
  ) -> WarehouseState:
    """
        Generate a warehouse in which the robot, boxes, and finish
        zones are connected through non-obstacle cells.
        """

    for attempt in range(maximum_attempts):
      candidate_seed = starting_seed + attempt
      state = self.simulation.reset(seed=candidate_seed)

      if self._all_targets_reachable(state):
        self.layout_seed = candidate_seed
        return state

    raise RuntimeError(
      "Unable to generate a reachable warehouse layout. "
      "Try fewer obstacles or a larger grid.")

  def _all_targets_reachable(
    self,
    state: WarehouseState,
  ) -> bool:
    """Check connectivity using breadth-first search."""

    visited = {state.robot}
    queue = deque([state.robot])

    movement_actions = (
      Action.UP,
      Action.RIGHT,
      Action.DOWN,
      Action.LEFT,
    )

    while queue:
      current = queue.popleft()

      for action in movement_actions:
        neighbour = current.moved(action)

        if not self.simulation.is_inside(neighbour):
          continue

        if neighbour in state.obstacle_positions:
          continue

        if neighbour in visited:
          continue

        visited.add(neighbour)
        queue.append(neighbour)

    required_positions = (set(state.boxes) | set(state.finish_zones))

    return required_positions.issubset(visited)

  def reset(
    self,
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[dict[str, np.ndarray | int], dict[str, Any]]:
    """
        Reset the episode to the same initial warehouse layout.

        The layout stays fixed during training so tabular Q-learning
        can learn a stable state-action mapping.
        """

    super().reset(seed=seed)

    self.robot_position = self.initial_state.robot
    self.boxes = set(self.initial_state.boxes)
    self.finish_zones = set(self.initial_state.finish_zones)
    self.used_finish_zones = set()

    self.carrying_box = False
    self.delivered_count = 0
    self.current_step = 0
    self.invalid_action_count = 0

    observation = self._get_observation()
    info = self._get_info()

    if self.render_mode == "ansi":
      print(self.render())

    return observation, info

  def step(
    self,
    action: int,
  ) -> tuple[
      dict[str, np.ndarray | int],
      float,
      bool,
      bool,
      dict[str, Any],
  ]:
    """Perform one robot action."""

    selected_action = Action(action)

    self.current_step += 1

    reward = -1.0

    if selected_action in (
        Action.UP,
        Action.RIGHT,
        Action.DOWN,
        Action.LEFT,
    ):
      reward = self._move_robot(selected_action)

    elif selected_action == Action.PICK_UP:
      reward = self._pick_up_box()

    elif selected_action == Action.DROP_OFF:
      reward = self._drop_off_box()

    terminated = (self.delivered_count == self.config.box_count)

    if terminated:
      reward += 100.0

    truncated = (self.current_step >= self.config.max_steps)

    observation = self._get_observation()
    info = self._get_info()

    if self.render_mode == "ansi":
      print(self.render())

    return (
      observation,
      reward,
      terminated,
      truncated,
      info,
    )

  def _move_robot(self, action: Action) -> float:
    """Move the robot if the destination is valid."""

    candidate_position = self.robot_position.moved(action)

    if not self.simulation.is_inside(candidate_position):
      self.invalid_action_count += 1
      return -5.0

    if (candidate_position in self.initial_state.obstacle_positions):
      self.invalid_action_count += 1
      return -5.0

    self.robot_position = candidate_position

    # Slightly larger reward when reaching a useful location.
    if (not self.carrying_box and self.robot_position in self.boxes):
      return 2.0

    if (self.carrying_box and self.robot_position in self.finish_zones
        and self.robot_position not in self.used_finish_zones):
      return 2.0

    return -1.0

  def _pick_up_box(self) -> float:
    """Pick up a box from the robot's current position."""

    if self.carrying_box:
      self.invalid_action_count += 1
      return -5.0

    if self.robot_position not in self.boxes:
      self.invalid_action_count += 1
      return -5.0

    self.boxes.remove(self.robot_position)
    self.carrying_box = True

    return 20.0

  def _drop_off_box(self) -> float:
    """Deliver a carried box to an unused finish zone."""

    if not self.carrying_box:
      self.invalid_action_count += 1
      return -5.0

    if self.robot_position not in self.finish_zones:
      self.invalid_action_count += 1
      return -5.0

    if self.robot_position in self.used_finish_zones:
      self.invalid_action_count += 1
      return -5.0

    self.carrying_box = False
    self.used_finish_zones.add(self.robot_position)
    self.delivered_count += 1

    return 50.0

  def _get_observation(self, ) -> dict[str, np.ndarray | int]:
    """Return the current Gymnasium observation."""

    boxes_array = np.zeros(
      self.config.width * self.config.height,
      dtype=np.int8,
    )

    finish_array = np.zeros(
      self.config.width * self.config.height,
      dtype=np.int8,
    )

    for box in self.boxes:
      index = box.row * self.config.width + box.column
      boxes_array[index] = 1

    for finish in self.finish_zones:
      index = (finish.row * self.config.width + finish.column)

      if finish not in self.used_finish_zones:
        finish_array[index] = 1

    return {
      "robot":
      np.array(
        [
          self.robot_position.row,
          self.robot_position.column,
        ],
        dtype=np.int64,
      ),
      "boxes":
      boxes_array,
      "finish_zones":
      finish_array,
      "carrying":
      int(self.carrying_box),
      "delivered":
      self.delivered_count,
    }

  def get_state_key(self) -> tuple:
    """
        Return a hashable state used by the Q-learning table.

        Obstacles do not need to appear here because the training
        layout remains fixed.
        """

    sorted_boxes = tuple(
      sorted((position.row, position.column) for position in self.boxes))

    sorted_used_destinations = tuple(
      sorted(
        (position.row, position.column)
        for position in self.used_finish_zones))

    return (
      self.robot_position.row,
      self.robot_position.column,
      int(self.carrying_box),
      sorted_boxes,
      sorted_used_destinations,
    )

  def _get_info(self) -> dict[str, Any]:
    """Return useful episode statistics."""

    return {
      "steps": self.current_step,
      "delivered": self.delivered_count,
      "remaining_boxes": len(self.boxes),
      "carrying": self.carrying_box,
      "invalid_actions": self.invalid_action_count,
      "layout_seed": self.layout_seed,
      "success": (self.delivered_count == self.config.box_count),
    }

  def render(self) -> str:
    """Return an ASCII representation of the current environment."""

    grid = np.full(
      (
        self.config.height,
        self.config.width,
      ),
      Tile.EMPTY,
      dtype=np.int8,
    )

    for position in self.initial_state.obstacle_positions:
      grid[position.row, position.column] = Tile.OBSTACLE

    for position in self.finish_zones:
      grid[position.row, position.column] = Tile.FINISH

    for position in self.used_finish_zones:
      grid[position.row, position.column] = Tile.EMPTY

    for position in self.boxes:
      grid[position.row, position.column] = Tile.BOX

    grid[
      self.robot_position.row,
      self.robot_position.column,
    ] = Tile.ROBOT

    symbols = {
      Tile.EMPTY: " . ",
      Tile.OBSTACLE: " # ",
      Tile.ROBOT: " R ",
      Tile.BOX: " B ",
      Tile.FINISH: " H ",
    }

    border = "+" + "---" * self.config.width + "+"

    lines = [
      border,
      *[
        "|" + "".join(symbols[Tile(cell)] for cell in row) + "|"
        for row in grid
      ],
      border,
      (
        f"Carrying: {self.carrying_box} | "
        f"Delivered: {self.delivered_count}/"
        f"{self.config.box_count} | "
        f"Steps: {self.current_step}"),
    ]

    return "\n".join(lines)
