import numpy as np
import pytest

from warehouse_robot.core.config import WarehouseConfig
from warehouse_robot.core.types import Position, Tile
from warehouse_robot.simulation.warehouse import (
  Obstacle,
  WarehouseSimulation,
  WarehouseState,
)


@pytest.fixture
def simulation() -> WarehouseSimulation:
  """
  Create a small warehouse with a known state.

  Layout:
    # # . . .
    # # . B .
    . . R . .
    . . . F .
    . . . . .
  """

  config = WarehouseConfig(
    width=5,
    height=5,
    obstacle_count=1,
    box_count=1,
    finish_count=1,
    max_steps=100,
    seed=42,
  )

  warehouse = WarehouseSimulation(config)

  obstacle = Obstacle(
    top_left=Position(row=0, column=0),
    width=2,
    height=2,
  )

  warehouse.state = WarehouseState(
    robot=Position(row=2, column=2),
    obstacles=(obstacle, ),
    obstacle_positions=obstacle.occupied_positions(),
    boxes=frozenset({
      Position(row=1, column=3),
    }),
    finish_zones=frozenset({
      Position(row=3, column=3),
    }),
  )

  return warehouse


def test_position_is_inside_warehouse(simulation: WarehouseSimulation) -> None:
  assert simulation.is_inside(Position(row=0, column=0))
  assert simulation.is_inside(Position(row=4, column=4))
  assert simulation.is_inside(Position(row=2, column=3))


def test_negative_position_is_outside_warehouse(
    simulation: WarehouseSimulation) -> None:
  assert not simulation.is_inside(Position(row=-1, column=0))
  assert not simulation.is_inside(Position(row=0, column=-1))


def test_position_beyond_boundary_is_outside(
    simulation: WarehouseSimulation) -> None:
  assert not simulation.is_inside(Position(row=5, column=0))
  assert not simulation.is_inside(Position(row=0, column=5))
  assert not simulation.is_inside(Position(row=10, column=10))


def test_is_obstacle(simulation: WarehouseSimulation) -> None:
  assert simulation.is_obstacle(Position(row=0, column=0))
  assert simulation.is_obstacle(Position(row=0, column=1))
  assert simulation.is_obstacle(Position(row=1, column=0))
  assert simulation.is_obstacle(Position(row=1, column=1))

  assert not simulation.is_obstacle(Position(row=2, column=2))
  assert not simulation.is_obstacle(Position(row=4, column=4))


def test_is_box(simulation: WarehouseSimulation) -> None:
  assert simulation.is_box(Position(row=1, column=3))
  assert not simulation.is_box(Position(row=2, column=2))
  assert not simulation.is_box(Position(row=3, column=3))


def test_is_finish_zone(simulation: WarehouseSimulation) -> None:
  assert simulation.is_finish_zone(Position(row=3, column=3))
  assert not simulation.is_finish_zone(Position(row=1, column=3))
  assert not simulation.is_finish_zone(Position(row=2, column=2))


def test_is_robot(simulation: WarehouseSimulation) -> None:
  assert simulation.is_robot(Position(row=2, column=2))
  assert not simulation.is_robot(Position(row=2, column=3))


def test_empty_position(simulation: WarehouseSimulation) -> None:
  assert simulation.is_empty(Position(row=4, column=4))
  assert simulation.is_empty(Position(row=2, column=3))


def test_obstacle_is_not_empty(simulation: WarehouseSimulation) -> None:
  assert not simulation.is_empty(Position(row=0, column=0))


def test_box_is_not_empty(simulation: WarehouseSimulation) -> None:
  assert not simulation.is_empty(Position(row=1, column=3))


def test_finish_zone_is_not_empty(simulation: WarehouseSimulation) -> None:
  assert not simulation.is_empty(Position(row=3, column=3))


def test_robot_position_is_not_empty(simulation: WarehouseSimulation) -> None:
  assert not simulation.is_empty(Position(row=2, column=2))


def test_outside_position_is_not_empty(
  simulation: WarehouseSimulation, ) -> None:
  assert not simulation.is_empty(Position(row=-1, column=0))
  assert not simulation.is_empty(Position(row=5, column=5))


def test_robot_can_enter_empty_position(
  simulation: WarehouseSimulation, ) -> None:
  assert simulation.can_robot_enter(Position(row=4, column=4))


def test_robot_can_enter_finish_zone(
  simulation: WarehouseSimulation, ) -> None:
  assert simulation.can_robot_enter(Position(row=3, column=3))


def test_robot_cannot_enter_obstacle(
  simulation: WarehouseSimulation, ) -> None:
  assert not simulation.can_robot_enter(Position(row=0, column=0))


def test_robot_cannot_enter_box(simulation: WarehouseSimulation, ) -> None:
  assert not simulation.can_robot_enter(Position(row=1, column=3))


def test_robot_cannot_leave_warehouse(
  simulation: WarehouseSimulation, ) -> None:
  assert not simulation.can_robot_enter(Position(row=-1, column=0))
  assert not simulation.can_robot_enter(Position(row=5, column=0))


def test_get_grid_returns_correct_dimensions(
  simulation: WarehouseSimulation, ) -> None:
  grid = simulation.get_grid()

  assert isinstance(grid, np.ndarray)
  assert grid.shape == (5, 5)


def test_get_grid_contains_correct_tiles(
  simulation: WarehouseSimulation, ) -> None:
  grid = simulation.get_grid()

  assert grid[0, 0] == Tile.OBSTACLE
  assert grid[0, 1] == Tile.OBSTACLE
  assert grid[1, 0] == Tile.OBSTACLE
  assert grid[1, 1] == Tile.OBSTACLE

  assert grid[1, 3] == Tile.BOX
  assert grid[2, 2] == Tile.ROBOT
  assert grid[3, 3] == Tile.FINISH
  assert grid[4, 4] == Tile.EMPTY


def test_two_by_two_obstacle_occupies_four_cells(
  simulation: WarehouseSimulation, ) -> None:
  assert simulation.state is not None
  assert len(simulation.state.obstacle_positions) == 4


def test_same_seed_produces_same_state() -> None:
  config = WarehouseConfig(
    width=15,
    height=15,
    obstacle_count=5,
    box_count=3,
    finish_count=3,
  )

  first_simulation = WarehouseSimulation(config)
  second_simulation = WarehouseSimulation(config)

  first_state = first_simulation.reset(seed=12345)
  second_state = second_simulation.reset(seed=12345)

  assert first_state == second_state

  np.testing.assert_array_equal(
    first_simulation.get_grid(),
    second_simulation.get_grid(),
  )


def test_generator_creates_requested_number_of_obstacles() -> None:
  config = WarehouseConfig(
    width=20,
    height=20,
    obstacle_count=5,
    box_count=3,
    finish_count=3,
  )

  warehouse = WarehouseSimulation(config)
  state = warehouse.reset(seed=42)

  assert len(state.obstacles) == 5


def test_generator_creates_all_supported_obstacle_shapes() -> None:
  config = WarehouseConfig(
    width=20,
    height=20,
    obstacle_count=5,
    box_count=3,
    finish_count=3,
  )

  warehouse = WarehouseSimulation(config)
  state = warehouse.reset(seed=42)

  generated_shapes = {
    (obstacle.width, obstacle.height)
    for obstacle in state.obstacles
  }

  expected_shapes = {
    (1, 1),
    (2, 2),
    (3, 2),
    (2, 3),
    (3, 3),
  }

  assert generated_shapes == expected_shapes


def test_generated_objects_do_not_overlap() -> None:
  config = WarehouseConfig(
    width=20,
    height=20,
    obstacle_count=5,
    box_count=4,
    finish_count=4,
  )

  warehouse = WarehouseSimulation(config)
  state = warehouse.reset(seed=42)

  assert state.robot not in state.obstacle_positions
  assert state.robot not in state.boxes
  assert state.robot not in state.finish_zones

  assert state.obstacle_positions.isdisjoint(state.boxes)
  assert state.obstacle_positions.isdisjoint(state.finish_zones)
  assert state.boxes.isdisjoint(state.finish_zones)


def test_methods_require_simulation_to_be_reset() -> None:
  config = WarehouseConfig(
    width=10,
    height=10,
    obstacle_count=1,
    box_count=1,
    finish_count=1,
  )

  warehouse = WarehouseSimulation(config)
  position = Position(row=1, column=1)

  with pytest.raises(RuntimeError, match=r"reset\(\)"):
    warehouse.is_obstacle(position)

  with pytest.raises(RuntimeError, match=r"reset\(\)"):
    warehouse.is_box(position)

  with pytest.raises(RuntimeError, match=r"reset\(\)"):
    warehouse.is_finish_zone(position)

  with pytest.raises(RuntimeError, match=r"reset\(\)"):
    warehouse.is_robot(position)

  with pytest.raises(RuntimeError, match=r"reset\(\)"):
    warehouse.is_empty(position)

  with pytest.raises(RuntimeError, match=r"reset\(\)"):
    warehouse.get_grid()
