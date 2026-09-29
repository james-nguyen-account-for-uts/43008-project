import pytest

from warehouse_robot.core.config import WarehouseConfig
from warehouse_robot.core.types import Action, Position
from warehouse_robot.environment import WarehouseEnv


@pytest.fixture
def environment() -> WarehouseEnv:
  config = WarehouseConfig(
    width=8,
    height=8,
    obstacle_count=4,
    box_count=1,
    finish_count=1,
    max_steps=50,
    seed=42,
  )

  return WarehouseEnv(
    config=config,
    layout_seed=42,
  )


def test_action_space_contains_all_actions(
  environment: WarehouseEnv, ) -> None:
  assert environment.action_space.n == len(Action)


def test_reset_returns_valid_observation(environment: WarehouseEnv, ) -> None:
  observation, info = environment.reset()

  assert environment.observation_space.contains(observation)

  assert info["steps"] == 0
  assert info["delivered"] == 0
  assert info["carrying"] is False
  assert info["success"] is False


def test_reset_restores_initial_robot_position(
  environment: WarehouseEnv, ) -> None:
  environment.reset()

  original_position = environment.robot_position
  environment.robot_position = Position(row=0, column=0)

  environment.reset()

  assert environment.robot_position == original_position


def test_reset_restores_all_boxes(environment: WarehouseEnv, ) -> None:
  environment.reset()

  original_boxes = set(environment.boxes)
  environment.boxes.clear()

  environment.reset()

  assert environment.boxes == original_boxes


def test_generated_layout_is_reachable(environment: WarehouseEnv, ) -> None:
  assert environment._all_targets_reachable(environment.initial_state)


def test_state_key_is_hashable(environment: WarehouseEnv, ) -> None:
  environment.reset()

  state_key = environment.get_state_key()

  # A hashable state can be used as a dictionary key.
  test_dictionary = {state_key: "valid"}

  assert test_dictionary[state_key] == "valid"


def test_state_key_changes_when_robot_moves(
  environment: WarehouseEnv, ) -> None:
  environment.reset()

  first_state = environment.get_state_key()

  original_position = environment.robot_position
  environment.robot_position = Position(
    row=(original_position.row + 1) % environment.config.height,
    column=original_position.column,
  )

  second_state = environment.get_state_key()

  assert first_state != second_state


def test_robot_cannot_move_outside_warehouse(
  environment: WarehouseEnv, ) -> None:
  environment.reset()
  environment.robot_position = Position(row=0, column=0)

  _, reward, terminated, truncated, info = (environment.step(Action.UP))

  assert environment.robot_position == Position(
    row=0,
    column=0,
  )
  assert reward == -5.0
  assert info["invalid_actions"] == 1
  assert terminated is False
  assert truncated is False


def test_robot_cannot_move_into_obstacle(environment: WarehouseEnv, ) -> None:
  environment.reset()

  obstacle_position = next(iter(environment.initial_state.obstacle_positions))

  possible_starts = (
    (
      Position(
        obstacle_position.row + 1,
        obstacle_position.column,
      ),
      Action.UP,
    ),
    (
      Position(
        obstacle_position.row - 1,
        obstacle_position.column,
      ),
      Action.DOWN,
    ),
    (
      Position(
        obstacle_position.row,
        obstacle_position.column + 1,
      ),
      Action.LEFT,
    ),
    (
      Position(
        obstacle_position.row,
        obstacle_position.column - 1,
      ),
      Action.RIGHT,
    ),
  )

  for starting_position, action in possible_starts:
    if not environment.simulation.is_inside(starting_position):
      continue

    if (starting_position in environment.initial_state.obstacle_positions):
      continue

    environment.robot_position = starting_position

    _, reward, _, _, info = environment.step(action)

    assert environment.robot_position == starting_position
    assert reward == -5.0
    assert info["invalid_actions"] == 1
    return

  pytest.skip("No accessible cell was found beside an obstacle.")


def test_robot_can_pick_up_box(environment: WarehouseEnv, ) -> None:
  environment.reset()

  box_position = next(iter(environment.boxes))
  environment.robot_position = box_position

  _, reward, terminated, truncated, info = (environment.step(Action.PICK_UP))

  assert reward == 20.0
  assert environment.carrying_box is True
  assert box_position not in environment.boxes
  assert info["carrying"] is True
  assert info["remaining_boxes"] == 0
  assert terminated is False
  assert truncated is False


def test_pickup_fails_when_robot_is_not_on_box(
  environment: WarehouseEnv, ) -> None:
  environment.reset()

  empty_position = next(
    Position(row=row, column=column)
    for row in range(environment.config.height)
    for column in range(environment.config.width)
    if Position(row=row, column=column) not in environment.initial_state.
    obstacle_positions and Position(row=row, column=column) not in environment.
    boxes and Position(row=row, column=column) not in environment.finish_zones)

  environment.robot_position = empty_position

  _, reward, terminated, _, info = environment.step(Action.PICK_UP)

  assert reward == -5.0
  assert environment.carrying_box is False
  assert info["invalid_actions"] == 1
  assert terminated is False


def test_robot_can_deliver_box(environment: WarehouseEnv, ) -> None:
  environment.reset()

  box_position = next(iter(environment.boxes))
  finish_position = next(iter(environment.finish_zones))

  environment.robot_position = box_position
  environment.step(Action.PICK_UP)

  environment.robot_position = finish_position

  _, reward, terminated, truncated, info = (environment.step(Action.DROP_OFF))

  # 50 for delivery and 100 for completing all deliveries.
  assert reward == 150.0
  assert environment.carrying_box is False
  assert environment.delivered_count == 1
  assert finish_position in environment.used_finish_zones
  assert info["success"] is True
  assert terminated is True
  assert truncated is False


def test_dropoff_fails_without_carried_box(
  environment: WarehouseEnv, ) -> None:
  environment.reset()

  finish_position = next(iter(environment.finish_zones))
  environment.robot_position = finish_position

  _, reward, terminated, _, info = environment.step(Action.DROP_OFF)

  assert reward == -5.0
  assert info["invalid_actions"] == 1
  assert info["delivered"] == 0
  assert terminated is False


def test_episode_is_truncated_at_maximum_steps() -> None:
  config = WarehouseConfig(
    width=8,
    height=8,
    obstacle_count=2,
    box_count=1,
    finish_count=1,
    max_steps=2,
    seed=42,
  )

  environment = WarehouseEnv(
    config=config,
    layout_seed=42,
  )

  environment.reset()
  environment.robot_position = Position(row=0, column=0)

  _, _, _, first_truncated, _ = environment.step(Action.UP)

  _, _, terminated, second_truncated, info = (environment.step(Action.UP))

  assert first_truncated is False
  assert second_truncated is True
  assert terminated is False
  assert info["steps"] == 2


def test_render_returns_warehouse_string(environment: WarehouseEnv, ) -> None:
  environment.reset()

  output = environment.render()

  assert isinstance(output, str)
  assert "R" in output
  assert "B" in output
  assert "H" in output
  assert "Delivered: 0/1" in output
