from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WarehouseConfig:
  """Values shared by the simulator, Gymnasium environment, and agents."""

  width: int = 10
  height: int = 10
  obstacle_count: int = 12
  box_count: int = 3
  finish_count: int = 3
  max_steps: int = 250
  seed: int | None = 42

  def __post_init__(self) -> None:
    if self.width < 3 or self.height < 3:
      raise ValueError("Warehouse dimensions must each be at least 3.")
    if min(self.obstacle_count, self.box_count, self.finish_count) < 0:
      raise ValueError("Object counts cannot be negative.")
    if self.box_count != self.finish_count:
      raise ValueError("Each box must have exactly one finish zone.")
    if self.max_steps <= 0:
      raise ValueError("max_steps must be positive.")

    required_cells = 1 + self.obstacle_count + self.box_count + self.finish_count
    if required_cells > self.width * self.height:
      raise ValueError(
        "The configured objects do not fit inside the warehouse grid.")
