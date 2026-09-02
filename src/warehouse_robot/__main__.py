from warehouse_robot.core.config import WarehouseConfig


def main() -> None:
  config = WarehouseConfig()
  print("Warehouse Robot RL foundation is ready.")
  print(f"Default grid: {config.width} x {config.height}")
  print(f"Maximum episode length: {config.max_steps} steps")


if __name__ == "__main__":
  main()
