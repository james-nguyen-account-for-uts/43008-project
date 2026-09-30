from __future__ import annotations

from typing import Protocol

import pygame

from warehouse_robot.core.types import Action
from warehouse_robot.environment import WarehouseEnv


class AgentProtocol(Protocol):
  """Minimum interface required by the visualiser."""

  def choose_action(
    self,
    state: tuple,
    training: bool = True,
  ) -> int:
    ...


class PygameVisualiser:
  """Graphical visualiser for the warehouse robot environment."""

  def __init__(
    self,
    environment: WarehouseEnv,
    cell_size: int = 42,
    side_panel_width: int = 300,
    fps: int = 5,
  ) -> None:

    self.environment = environment
    self.cell_size = cell_size
    self.side_panel_width = side_panel_width
    self.fps = fps

    self.grid_width = (
      environment.config.width * self.cell_size
    )

    self.grid_height = (
      environment.config.height * self.cell_size
    )

    self.window_width = (
      self.grid_width + self.side_panel_width
    )

    self.window_height = max(
      self.grid_height,
      600,
    )

    pygame.init()

    pygame.display.set_caption(
      "Warehouse Robot - Q-Learning Visualisation"
    )

    self.screen = pygame.display.set_mode(
      (
        self.window_width,
        self.window_height,
      )
    )

    self.clock = pygame.time.Clock()

    self.font = pygame.font.SysFont(
      "arial",
      21,
    )

    self.small_font = pygame.font.SysFont(
      "arial",
      17,
    )

    self.large_font = pygame.font.SysFont(
      "arial",
      28,
      bold=True,
    )

    self.running = True

  # --------------------------------------------------
  # Public methods
  # --------------------------------------------------

  def run_trained_policy(
    self,
    agent: AgentProtocol,
  ) -> dict[str, object]:
    """
    Run one episode using the trained greedy policy.

    The visualiser displays every action performed by the robot.
    """

    _, info = self.environment.reset()

    state = self.environment.get_state_key()

    terminated = False
    truncated = False

    total_reward = 0.0
    last_reward = 0.0
    last_action: Action | None = None

    self.draw(
      last_action=None,
      reward=0.0,
      total_reward=0.0,
      terminated=False,
      truncated=False,
    )

    self._wait_for_next_frame()

    while (
      not terminated
      and not truncated
      and self.running
    ):

      self._handle_events()

      if not self.running:
        break

      action_value = agent.choose_action(
        state,
        training=False,
      )

      last_action = Action(action_value)

      (
        _,
        last_reward,
        terminated,
        truncated,
        info,
      ) = self.environment.step(action_value)

      total_reward += last_reward

      state = self.environment.get_state_key()

      self.draw(
        last_action=last_action,
        reward=last_reward,
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
      )

      self._wait_for_next_frame()

    # Keep final state visible for a short period.
    if self.running:
      final_wait_frames = self.fps * 3

      for _ in range(final_wait_frames):
        self._handle_events()

        if not self.running:
          break

        self.clock.tick(self.fps)

    result = {
      "success": info.get("success", False),
      "steps": info.get("steps", 0),
      "delivered": info.get("delivered", 0),
      "invalid_actions": info.get(
        "invalid_actions",
        0,
      ),
      "total_reward": total_reward,
    }

    self.close()

    return result

  def draw(
    self,
    last_action: Action | None = None,
    reward: float = 0.0,
    total_reward: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
  ) -> None:
    """Draw the complete warehouse and information panel."""

    self.screen.fill((245, 245, 245))

    self._draw_grid()
    self._draw_finish_zones()
    self._draw_obstacles()
    self._draw_boxes()
    self._draw_robot()

    self._draw_information_panel(
      last_action=last_action,
      reward=reward,
      total_reward=total_reward,
      terminated=terminated,
      truncated=truncated,
    )

    pygame.display.flip()

  def close(self) -> None:
    """Close the Pygame window."""

    pygame.quit()

  # --------------------------------------------------
  # Warehouse drawing
  # --------------------------------------------------

  def _draw_grid(self) -> None:
    """Draw warehouse grid cells."""

    grid_colour = (205, 205, 205)

    for row in range(
      self.environment.config.height + 1
    ):
      y = row * self.cell_size

      pygame.draw.line(
        self.screen,
        grid_colour,
        (0, y),
        (self.grid_width, y),
        1,
      )

    for column in range(
      self.environment.config.width + 1
    ):
      x = column * self.cell_size

      pygame.draw.line(
        self.screen,
        grid_colour,
        (x, 0),
        (x, self.grid_height),
        1,
      )

  def _draw_obstacles(self) -> None:
    """Draw obstacle cells."""

    obstacle_colour = (70, 70, 70)

    for position in (
      self.environment.initial_state.obstacle_positions
    ):
      rectangle = self._cell_rectangle(
        position.row,
        position.column,
      )

      pygame.draw.rect(
        self.screen,
        obstacle_colour,
        rectangle,
      )

      pygame.draw.rect(
        self.screen,
        (40, 40, 40),
        rectangle,
        2,
      )

  def _draw_finish_zones(self) -> None:
    """Draw unused and completed finish zones."""

    for position in self.environment.finish_zones:

      rectangle = self._cell_rectangle(
        position.row,
        position.column,
      )

      if position in (
        self.environment.used_finish_zones
      ):
        colour = (170, 220, 170)

        pygame.draw.rect(
          self.screen,
          colour,
          rectangle,
        )

        self._draw_centered_text(
          "✓",
          rectangle,
          (40, 110, 40),
          self.large_font,
        )

      else:
        colour = (120, 205, 130)

        pygame.draw.rect(
          self.screen,
          colour,
          rectangle,
        )

        pygame.draw.rect(
          self.screen,
          (30, 130, 50),
          rectangle,
          3,
        )

        self._draw_centered_text(
          "F",
          rectangle,
          (20, 100, 40),
          self.font,
        )

  def _draw_boxes(self) -> None:
    """Draw all undelivered boxes."""

    box_colour = (205, 145, 70)
    border_colour = (120, 75, 30)

    padding = 7

    for position in self.environment.boxes:

      outer_rectangle = self._cell_rectangle(
        position.row,
        position.column,
      )

      rectangle = pygame.Rect(
        outer_rectangle.x + padding,
        outer_rectangle.y + padding,
        outer_rectangle.width - 2 * padding,
        outer_rectangle.height - 2 * padding,
      )

      pygame.draw.rect(
        self.screen,
        box_colour,
        rectangle,
        border_radius=3,
      )

      pygame.draw.rect(
        self.screen,
        border_colour,
        rectangle,
        2,
        border_radius=3,
      )

      # Box cross detail
      pygame.draw.line(
        self.screen,
        border_colour,
        rectangle.topleft,
        rectangle.bottomright,
        2,
      )

      pygame.draw.line(
        self.screen,
        border_colour,
        rectangle.topright,
        rectangle.bottomleft,
        2,
      )

  def _draw_robot(self) -> None:
    """Draw the robot at its current position."""

    position = self.environment.robot_position

    rectangle = self._cell_rectangle(
      position.row,
      position.column,
    )

    centre = rectangle.center

    radius = max(
      10,
      self.cell_size // 3,
    )

    robot_colour = (70, 130, 220)

    if self.environment.carrying_box:
      robot_colour = (130, 90, 210)

    pygame.draw.circle(
      self.screen,
      robot_colour,
      centre,
      radius,
    )

    pygame.draw.circle(
      self.screen,
      (30, 60, 120),
      centre,
      radius,
      2,
    )

    # Robot eyes
    eye_y = centre[1] - 4

    pygame.draw.circle(
      self.screen,
      (255, 255, 255),
      (
        centre[0] - 6,
        eye_y,
      ),
      3,
    )

    pygame.draw.circle(
      self.screen,
      (255, 255, 255),
      (
        centre[0] + 6,
        eye_y,
      ),
      3,
    )

    # Show small box indicator while carrying.
    if self.environment.carrying_box:
      box_size = max(
        8,
        self.cell_size // 5,
      )

      box_rectangle = pygame.Rect(
        centre[0] - box_size // 2,
        centre[1] + 5,
        box_size,
        box_size,
      )

      pygame.draw.rect(
        self.screen,
        (230, 170, 80),
        box_rectangle,
      )

  # --------------------------------------------------
  # Information panel
  # --------------------------------------------------

  def _draw_information_panel(
    self,
    last_action: Action | None,
    reward: float,
    total_reward: float,
    terminated: bool,
    truncated: bool,
  ) -> None:

    panel_x = self.grid_width

    pygame.draw.rect(
      self.screen,
      (235, 238, 242),
      (
        panel_x,
        0,
        self.side_panel_width,
        self.window_height,
      ),
    )

    pygame.draw.line(
      self.screen,
      (180, 180, 180),
      (panel_x, 0),
      (panel_x, self.window_height),
      2,
    )

    x = panel_x + 20
    y = 25

    self._draw_text(
      "Q-Learning Robot",
      x,
      y,
      self.large_font,
      (30, 30, 30),
    )

    y += 55

    action_name = (
      last_action.name
      if last_action is not None
      else "START"
    )

    information = [
      f"Step: {self.environment.current_step}",
      f"Action: {action_name}",
      f"Reward: {reward:.1f}",
      f"Total reward: {total_reward:.1f}",
      "",
      (
        "Carrying box: "
        f"{self.environment.carrying_box}"
      ),
      (
        "Delivered: "
        f"{self.environment.delivered_count}"
        f"/{self.environment.config.box_count}"
      ),
      (
        "Remaining boxes: "
        f"{len(self.environment.boxes)}"
      ),
      (
        "Invalid actions: "
        f"{self.environment.invalid_action_count}"
      ),
    ]

    for line in information:

      if line == "":
        y += 15
        continue

      self._draw_text(
        line,
        x,
        y,
        self.font,
        (45, 45, 45),
      )

      y += 34

    y += 15

    if terminated:
      self._draw_text(
        "SUCCESS",
        x,
        y,
        self.large_font,
        (30, 140, 60),
      )

      y += 38

      self._draw_text(
        "All boxes delivered.",
        x,
        y,
        self.small_font,
        (30, 110, 50),
      )

    elif truncated:
      self._draw_text(
        "STEP LIMIT REACHED",
        x,
        y,
        self.font,
        (190, 70, 50),
      )

    else:
      self._draw_text(
        "Running...",
        x,
        y,
        self.font,
        (60, 100, 170),
      )

    # Legend
    legend_y = self.window_height - 170

    self._draw_text(
      "Legend",
      x,
      legend_y,
      self.font,
      (30, 30, 30),
    )

    legend_y += 30

    legend = [
      "Blue circle = Robot",
      "Brown box = Package",
      "Green = Finish zone",
      "Dark grey = Obstacle",
      "Purple robot = Carrying",
    ]

    for line in legend:
      self._draw_text(
        line,
        x,
        legend_y,
        self.small_font,
        (70, 70, 70),
      )

      legend_y += 24

  # --------------------------------------------------
  # Helpers
  # --------------------------------------------------

  def _cell_rectangle(
    self,
    row: int,
    column: int,
  ) -> pygame.Rect:

    return pygame.Rect(
      column * self.cell_size,
      row * self.cell_size,
      self.cell_size,
      self.cell_size,
    )

  def _draw_text(
    self,
    text: str,
    x: int,
    y: int,
    font: pygame.font.Font,
    colour: tuple[int, int, int],
  ) -> None:

    surface = font.render(
      text,
      True,
      colour,
    )

    self.screen.blit(
      surface,
      (x, y),
    )

  def _draw_centered_text(
    self,
    text: str,
    rectangle: pygame.Rect,
    colour: tuple[int, int, int],
    font: pygame.font.Font,
  ) -> None:

    surface = font.render(
      text,
      True,
      colour,
    )

    text_rectangle = surface.get_rect(
      center=rectangle.center,
    )

    self.screen.blit(
      surface,
      text_rectangle,
    )

  def _handle_events(self) -> None:
    """Handle Pygame window events."""

    for event in pygame.event.get():

      if event.type == pygame.QUIT:
        self.running = False

      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          self.running = False

  def _wait_for_next_frame(self) -> None:
    """Wait according to configured visualisation FPS."""

    self._handle_events()

    self.clock.tick(self.fps)