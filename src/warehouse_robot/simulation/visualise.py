import secrets
import argparse

from warehouse_robot.core.config import WarehouseConfig
from warehouse_robot.core.types import Tile
from warehouse_robot.simulation import WarehouseSimulation

TILE_SYMBOLS = {
  Tile.EMPTY: " . ",
  Tile.OBSTACLE: " # ",
  Tile.ROBOT: " R ",
  Tile.BOX: " B ",
  Tile.FINISH: " F ",
}


def render_terminal(simulation: WarehouseSimulation) -> None:
  """Print the generated warehouse as an ASCII image."""

  grid = simulation.get_grid()

  horizontal_border = "+" + "---" * simulation.config.width + "+"

  print()
  print(horizontal_border)

  for row in grid:
    row_symbols = "".join(TILE_SYMBOLS[Tile(cell)] for cell in row)

    print(f"|{row_symbols}|")

  print(horizontal_border)
  print()

  print("Legend:")
  print("  R = Robot")
  print("  B = Box")
  print("  F = Finish zone")
  print("  # = Obstacle")
  print("  . = Empty space")


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Generate and display a random warehouse.")

  parser.add_argument(
    "--seed",
    type=int,
    # default=42,
    # help="Random seed used to generate the warehouse.",
    default=None,
    help=(
      "Optional random seed. If omitted, a new seed is generated every time."),
  )

  parser.add_argument(
    "--width",
    type=int,
    default=15,
    help="Warehouse width.",
  )

  parser.add_argument(
    "--height",
    type=int,
    default=15,
    help="Warehouse height.",
  )

  parser.add_argument(
    "--obstacles",
    type=int,
    default=10,
    help="Number of obstacles.",
  )

  parser.add_argument(
    "--boxes",
    type=int,
    default=3,
    help="Number of boxes and finish zones.",
  )

  return parser.parse_args()


def main() -> None:
  arguments = parse_arguments()

  # Generate a new unpredictable seed when one is not provided.
  selected_seed = (
    arguments.seed if arguments.seed is not None else secrets.randbits(32))

  config = WarehouseConfig(
    width=arguments.width,
    height=arguments.height,
    obstacle_count=arguments.obstacles,
    box_count=arguments.boxes,
    finish_count=arguments.boxes,
    seed=selected_seed,
  )

  simulation = WarehouseSimulation(config)
  state = simulation.reset()

  print("Warehouse Robot Simulation")
  print(f"Seed: {selected_seed}")
  print(f"Size: {arguments.width} x {arguments.height}")

  render_terminal(simulation)


if __name__ == "__main__":
  main()
