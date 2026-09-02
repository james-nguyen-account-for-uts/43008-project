import pytest

from warehouse_robot.core.config import WarehouseConfig


def test_default_configuration_is_valid() -> None:
  config = WarehouseConfig()
  assert config.width == 15
  assert config.box_count == config.finish_count


def test_boxes_need_matching_finish_zones() -> None:
  with pytest.raises(ValueError, match="exactly one finish zone"):
    WarehouseConfig(box_count=2, finish_count=1)


def test_objects_must_fit_in_grid() -> None:
  with pytest.raises(ValueError, match="do not fit"):
    WarehouseConfig(
      width=3, height=3, obstacle_count=8, box_count=1, finish_count=1)
