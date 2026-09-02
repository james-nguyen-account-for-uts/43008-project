from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
  """Discrete actions that the future Gymnasium environment will expose."""
  UP = 0
  RIGHT = 1
  DOWN = 2
  LEFT = 3
  PICK_UP = 4
  DROP_OFF = 5


class Tile(IntEnum):
  """Integer tile values suitable for NumPy observations."""
  EMPTY = 0
  OBSTACLE = 1
  ROBOT = 2
  BOX = 3
  FINISH = 4


@dataclass(frozen=True, slots=True)
class Position:
  row: int
  column: int

  def moved(self, action: Action) -> "Position":
    offsets = {
      Action.UP: (-1, 0),
      Action.RIGHT: (0, 1),
      Action.DOWN: (1, 0),
      Action.LEFT: (0, -1),
    }
    if action not in offsets:
      return self
    row_change, column_change = offsets[action]
    return Position(self.row + row_change, self.column + column_change)
