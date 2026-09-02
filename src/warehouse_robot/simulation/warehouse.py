from __future__ import annotations

import random
import numpy as np
from dataclasses import dataclass

from warehouse_robot.core.config import WarehouseConfig
from warehouse_robot.core.types import Tile, Position

OBSTACLE_SHAPES = ((1, 1), (2, 2), (3, 2), (2, 3), (3, 3))


@dataclass(frozen=True, slots=True)
class Obstacle:
  """
  Represents a rectangular obstacle in the warehouse.
  
  The obstacle is defined by its top-left position and its width and height.
  List of shapes available for obstacles: 1x1, 2x2, 3x2, 2x3, 3x3.
  """

  top_left: Position
  width: int
  height: int

  def occupied_positions(self) -> frozenset[Position]:
    return frozenset(
      Position(
        row=self.top_left.row + row_offset,
        column=self.top_left.column + column_offset)
      for row_offset in range(self.height)
      for column_offset in range(self.width))


@dataclass(frozen=True, slots=True)
class WarehouseState:
  """Contains the generated positions of all warehouse objects."""

  robot: Position
  obstacles: tuple[Obstacle, ...]
  obstacle_positions: frozenset[Position]
  boxes: frozenset[Position]
  finish_zones: frozenset[Position]


class WarehouseSimulation:
  """
  Generates and stores a random warehouse environment.

  The simulation currently handles environment generation only.
  Robot movement and box interaction can be added afterwards.
  """

  def __init__(self, config: WarehouseConfig) -> None:
    self.config = config
    self.state: WarehouseState | None = None

  def reset(self, seed: int | None = None) -> WarehouseState:
    selected_seed = self.config.seed if seed is None else seed
    random_generator = random.Random(selected_seed)

    # Returns rectangle objects and their individual occupied cells
    obstacles, obstacle_positions = self._generate_obstacles(random_generator)

    # Check if there is enough space for the robot, boxes, and finish zones
    available_positions = [
      Position(row=row, column=column) for row in range(self.config.height)
      for column in range(self.config.width)
      if Position(row=row, column=column) not in obstacle_positions
    ]
    required_positions = (1 + self.config.box_count + self.config.finish_count)
    if len(available_positions) < required_positions:
      raise ValueError(
        "The obstacles leave insufficient space for the robot, boxes, and finish zones."
      )

    random_generator.shuffle(available_positions)

    current_index = 0

    robot = available_positions[current_index]
    current_index += 1

    boxes = frozenset(
      available_positions[current_index:current_index + self.config.box_count])
    current_index += self.config.box_count

    finish_zones = frozenset(
      available_positions[current_index:current_index +
                          self.config.finish_count])

    self.state = WarehouseState(
      robot=robot,
      obstacles=obstacles,
      obstacle_positions=obstacle_positions,
      boxes=boxes,
      finish_zones=finish_zones,
    )
    return self.state

  def _generate_obstacles(
    self, random_generator: random.Random
  ) -> tuple[tuple[Obstacle, ...], frozenset[Position]]:
    """
    Generate non-overlapping rectangular obstacles.

    Shape selection uses a shuffled bag. Therefore, every group
    of five obstacles contains one of each available shape.
    """

    obstacles: list[Obstacle] = []
    occupied_positions: set[Position] = set()

    shape_bag = list(OBSTACLE_SHAPES)
    random_generator.shuffle(shape_bag)

    maximum_attempts_per_obstacle = 200

    for obstacle_number in range(self.config.obstacle_count):
      # Refill and reshuffle the shape bag after using every shape.
      if obstacle_number > 0 and obstacle_number % len(shape_bag) == 0:
        random_generator.shuffle(shape_bag)

      width, height = shape_bag[obstacle_number % len(shape_bag)]

      obstacle_placed = False

      for _ in range(maximum_attempts_per_obstacle):
        if (width > self.config.width or height > self.config.height):
          break

        maximum_row = self.config.height - height
        maximum_column = self.config.width - width

        top_left = Position(
          row=random_generator.randint(0, maximum_row),
          column=random_generator.randint(0, maximum_column),
        )

        candidate = Obstacle(
          top_left=top_left,
          width=width,
          height=height,
        )

        candidate_positions = candidate.occupied_positions()

        # The candidate must not overlap an existing obstacle.
        if candidate_positions & occupied_positions:
          continue

        obstacles.append(candidate)
        occupied_positions.update(candidate_positions)

        obstacle_placed = True
        break

      if not obstacle_placed:
        raise RuntimeError(
          f"Unable to place a {width}x{height} obstacle. "
          "Try using fewer obstacles or a larger warehouse.")

    return (
      tuple(obstacles),
      frozenset(occupied_positions),
    )

  def get_grid(self) -> np.ndarray:
    """
    Convert the current warehouse state into a two-dimensional NumPy array.

    Each position contains an integer from the Tile enum.
    """

    if self.state is None:
      raise RuntimeError(
        "The simulation has not been generated. Call reset() before requesting the grid."
      )

    grid = np.full(
      shape=(self.config.height, self.config.width),
      fill_value=Tile.EMPTY,
      dtype=np.int8,
    )

    for position in self.state.obstacle_positions:
      grid[position.row, position.column] = Tile.OBSTACLE

    for position in self.state.finish_zones:
      grid[position.row, position.column] = Tile.FINISH

    for position in self.state.boxes:
      grid[position.row, position.column] = Tile.BOX

    robot = self.state.robot
    grid[robot.row, robot.column] = Tile.ROBOT

    return grid

  def is_inside(self, position: Position) -> bool:
    """Return True if a position is inside the warehouse."""

    return (
      0 <= position.row < self.config.height
      and 0 <= position.column < self.config.width)

  def is_obstacle(self, position: Position) -> bool:
    """Return True if a position is occupied by an obstacle."""

    if self.state is None:
      raise RuntimeError("Call reset() before checking the warehouse.")

    return position in self.state.obstacle_positions

  def is_box(self, position: Position) -> bool:
    """Return True if a position contains an undelivered box."""

    if self.state is None:
      raise RuntimeError("Call reset() before checking the warehouse.")

    return position in self.state.boxes

  def is_finish_zone(self, position: Position) -> bool:
    """Return True if a position contains a finish zone."""

    if self.state is None:
      raise RuntimeError("Call reset() before checking the warehouse.")

    return position in self.state.finish_zones

  def is_robot(self, position: Position) -> bool:
    """Return True if the robot is at the given position."""

    if self.state is None:
      raise RuntimeError("Call reset() before checking the warehouse.")

    return position == self.state.robot

  def is_empty(self, position: Position) -> bool:
    """Return True if the position is inside the warehouse and does not contain an obstacle, box, finish zone, or robot."""

    if self.state is None:
      raise RuntimeError("Call reset() before checking the warehouse.")

    return (
      self.is_inside(position) and not self.is_obstacle(position)
      and not self.is_box(position) and not self.is_finish_zone(position)
      and not self.is_robot(position))

  def can_robot_enter(self, position: Position) -> bool:
    """
    Return True if the robot can enter the position.

    Finish zones are enterable. Obstacles and boxes currently block movement until box interaction is implemented.
    """

    return (
      self.is_inside(position) and not self.is_obstacle(position)
      and not self.is_box(position))
