from warehouse_robot.core.types import Action, Position


def test_position_moves_in_four_directions() -> None:
  start = Position(2, 2)
  assert start.moved(Action.UP) == Position(1, 2)
  assert start.moved(Action.RIGHT) == Position(2, 3)
  assert start.moved(Action.DOWN) == Position(3, 2)
  assert start.moved(Action.LEFT) == Position(2, 1)


def test_non_movement_action_keeps_position() -> None:
  start = Position(2, 2)
  assert start.moved(Action.PICK_UP) == start
