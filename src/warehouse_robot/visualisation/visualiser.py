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

    # Store the warehouse environment and the main settings
    self.environment = environment
    self.cell_size = cell_size
    self.side_panel_width = side_panel_width
    self.fps = fps

    # Calculate grid and window dimensions
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

    # Initialise Pygame and create the window
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

 
  def run_trained_policy(
    self,
    agent: AgentProtocol,
  ) -> dict[str, object]:
    """
    Run one episode using the trained greedy policy.
    The visualiser displays every action performed by the robot.
    """

    # Reset the environment and initialise episode variables.
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

    # Run the trained policy until the episode ends.
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
        last_action=None,
        reward=0.0,
        total_reward=0.0,
        terminated=False,
        truncated=False,
      )

      # Wait for the Start button before running the simulation.
      self._wait_for_start()

      # Run the trained policy
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
      final_wait_frames = self.fps * 5

      for _ in range(final_wait_frames):
        self._handle_events()

        if not self.running:
          break

        self.clock.tick(self.fps)

    # Collect episode statistics and close the visualiser.
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

    # Clear the screen and draw all warehouse elements.
    self.screen.fill((245, 245, 245))

    self._draw_grid()
    self._draw_finish_zones()
    self._draw_obstacles()
    self._draw_boxes()
    self._draw_robot()

    # Display the current action, rewards, and episode status.
    self._draw_information_panel(
      last_action=last_action,
      reward=reward,
      total_reward=total_reward,
      terminated=terminated,
      truncated=truncated,
    )

    # Update the display to show the latest frame.
    pygame.display.flip()

  def close(self) -> None:
    """Close the Pygame window."""

    pygame.quit()

  
  # Warehouse drawing
  def _draw_grid(self) -> None:
    """Draw warehouse grid cells."""

    # Set the grid line colour.
    grid_colour = (205, 205, 205)

    # Draw horizontal grid lines.
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

    # Draw vertical grid lines.
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

    # Set the obstacle colour.
    obstacle_colour = (71, 85, 105)
    border_colour = (52, 65, 85)
    highlight_colour = (96, 114, 140)

    # Get obstacle positions from the environment.
    for position in (
      self.environment.initial_state.obstacle_positions
    ):
      rectangle = self._cell_rectangle(
        position.row,
        position.column,
      )

      # Add spacing between obstacle cells.
      inner_rect = rectangle.inflate(-4, -4)

      # Draw rounded obstacle blocks.
      pygame.draw.rect(
        self.screen,
        obstacle_colour,
        inner_rect,
        border_radius=4,
      )

      # Draw the obstacle border.
      pygame.draw.rect(
        self.screen,
        border_colour,
        inner_rect,
        2,
        border_radius=4,
      )

      # Add a subtle highlight.
      pygame.draw.line(
        self.screen,
        highlight_colour,
        (inner_rect.x + 4, inner_rect.y + 5),
        (inner_rect.right - 5, inner_rect.y + 5),
        2,
      )

  def _draw_finish_zones(self) -> None:
    """Draw unused and completed finish zones."""

    # Get finish zone positions from the environment.
    for position in self.environment.finish_zones:

      rectangle = self._cell_rectangle(
        position.row,
        position.column,
      )

      # Display completed finish zones with a DONE.
      if position in (
        self.environment.used_finish_zones
      ):
        colour = (170, 220, 170)

        pygame.draw.rect(
          self.screen,
          colour,
          rectangle,
        )

        done_font = pygame.font.SysFont("arial", 12, bold=True)

        self._draw_centered_text(
          "DONE",
          rectangle,
          (40, 110, 40),
          done_font,
        )

      # Display unused finish zones.
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

        pygame.draw.circle(
          self.screen,
          (32, 99, 64),
          rectangle.center,
          10,
          2,
        )

        pygame.draw.circle(
          self.screen,
          (32, 99, 64),
          rectangle.center,
          3,
        )


  def _draw_boxes(self) -> None:
    """Draw all undelivered boxes."""

    box_colour = (205, 145, 70)
    border_colour = (120, 75, 30)
    tape_colour = (239, 212, 161)
    label_colour = (244, 238, 227)

    padding = 5

    for position in self.environment.boxes:

      # Calculate the box position and size.
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

      # Draw the cardboard box.
      pygame.draw.rect(
        self.screen,
        box_colour,
        rectangle,
        border_radius=3,
      )

      # Draw packing tape across the centre.
      tape_width = max(4, rectangle.width // 5)
      tape_rectangle = pygame.Rect(
        rectangle.centerx - tape_width // 2,
        rectangle.y + 2,
        tape_width,
        rectangle.height - 4,
      )

      pygame.draw.rect(
        self.screen,
        tape_colour,
        tape_rectangle,
      )

      # Add a shipping label.
      label_rectangle = pygame.Rect(
        rectangle.x + 3,
        rectangle.y + rectangle.height // 2,
        max(6, rectangle.width // 3),
        max(7, rectangle.height // 3),
      )

      pygame.draw.rect(
        self.screen,
        label_colour,
        label_rectangle,
        border_radius=1,
      )

      # Add small lines to represent label details.
      for i in range(2):
        y = label_rectangle.y + 3 + i * 3
        pygame.draw.line(
          self.screen,
          (130, 120, 105),
          (label_rectangle.x + 2, y),
          (label_rectangle.right - 2, y),
          1,
        )

      # Draw the box border.
      pygame.draw.rect(
        self.screen,
        border_colour,
        rectangle,
        2,
        border_radius=3,
      )

  def _draw_robot(self) -> None:
    """Draw a simple warehouse robot."""

    position = self.environment.robot_position
    rect = self._cell_rectangle(
      position.row,
      position.column,
    )
    cx, cy = rect.center

    # Change colour when carrying a box.
    colour = (130, 90, 210) if self.environment.carrying_box else (70, 130, 220)

    # Draw wheels.
    pygame.draw.rect(
      self.screen,
      (55, 65, 80),
      (cx - 16, cy - 8, 6, 16),
    )
    pygame.draw.rect(
      self.screen,
      (55, 65, 80),
      (cx + 10, cy - 8, 6, 16),
    )

    # Draw body.
    pygame.draw.rect(
      self.screen,
      colour,
      (cx - 13, cy - 13, 26, 26),
      border_radius=5,
    )

    pygame.draw.rect(
      self.screen,
      (30, 60, 120),
      (cx - 13, cy - 13, 26, 26),
      2,
      border_radius=5,
    )

    # Draw sensor bar.
    pygame.draw.rect(
      self.screen,
      (140, 235, 245),
      (cx - 7, cy - 8, 14, 5),
      border_radius=2,
    )

    # Show small box indicator while carrying.
    if self.environment.carrying_box:
      box_rectangle = pygame.Rect(
        cx - 5,
        cy + 2,
        10,
        8,
      )

      pygame.draw.rect(
        self.screen,
        (230, 170, 80),
        box_rectangle,
      )

  # Information panel
  def _draw_information_panel(
    self,
    last_action: Action | None,
    reward: float,
    total_reward: float,
    terminated: bool,
    truncated: bool,
  ) -> None:

    # Draw the information panel background and border.
    panel_x = self.grid_width
    panel_rect = pygame.Rect(
      panel_x,
      0,
      self.side_panel_width,
      self.window_height,
    )

    # Draw the panel background.
    pygame.draw.rect(
      self.screen,
      (240, 243, 248),
      panel_rect,
    )

    pygame.draw.line(
      self.screen,
      (210, 214, 222),
      (panel_x, 0),
      (panel_x, self.window_height),
      2,
    )

    x = panel_x + 18
    y = 18
    card_width = self.side_panel_width - 36

    # Display the panel title.
    self._draw_text(
      "Warehouse Robot",
      x,
      y,
      self.large_font,
      (25, 30, 40),
    )

    # Simulation Info Card
    info_card = pygame.Rect(x, 70, card_width, 265)

    pygame.draw.rect(
      self.screen,
      (250, 251, 253),
      info_card,
      border_radius=14,
    )

    # section title
    self._draw_text(
      "Simulation Info",
      x + 14,
      84,
      self.font,
      (55, 85, 150),
    )

    action_name = (
      last_action.name if last_action is not None else "START"
    )

    info_rows = [
      ("Step", str(self.environment.current_step)),
      ("Action", action_name),
      ("Reward", f"{reward:.1f}"),
      ("Total reward", f"{total_reward:.1f}"),
      ("divider", ""),
      ("Carrying box", str(self.environment.carrying_box)),
      (
        "Delivered",
        f"{self.environment.delivered_count} / {self.environment.config.box_count}",),
      ("Remaining boxes", str(len(self.environment.boxes))),
      ("Invalid actions", str(self.environment.invalid_action_count)),
    ]

    row_y = 116

    for label, value in info_rows:

      if label == "divider":
        pygame.draw.line(
          self.screen,
          (225, 228, 235),
          (x + 12, row_y),
          (x + card_width - 12, row_y),
          1,
        )
        row_y += 12
        continue

      self._draw_text(
        label,
        x + 14,
        row_y,
        self.small_font,
        (75, 78, 86),
      )

      # Action pill
      if label == "Action":
        pill_rect = pygame.Rect(
          x + card_width - 118,
          row_y - 2,
          96,
          22,
        )

        pygame.draw.rect(
          self.screen,
          (214, 231, 255),
          pill_rect,
          border_radius=11,
        )

        self._draw_centered_text(
          value,
          pill_rect,
          (66, 120, 210),
          self.small_font,
        )

      # Carrying box pill
      elif label == "Carrying box":
        pill_rect = pygame.Rect(
          x + card_width - 72,
          row_y - 2,
          50,
          22,
        )

        pygame.draw.rect(
          self.screen,
          (235, 237, 242),
          pill_rect,
          border_radius=11,
        )

        self._draw_centered_text(
          value,
          pill_rect,
          (125, 130, 140),
          self.small_font,
        )

      else:
        value_colour = (70, 72, 80)

        if label == "Reward" and reward < 0:
          value_colour = (220, 85, 75)

        value_surface = self.small_font.render(
          value,
          True,
          value_colour,
        )

        self.screen.blit(
          value_surface,
          (
            x + card_width - 20 - value_surface.get_width(),
            row_y,
          ),
        )

      row_y += 25

    # status card
    status_card = pygame.Rect(x, 352, card_width, 110)

    pygame.draw.rect(
      self.screen,
      (250, 251, 253),
      status_card,
      border_radius=14,
    )

    self._draw_text(
      "Status",
      x + 14,
      366,
      self.font,
      (55, 85, 150),
    )

    if terminated:
      status_text = "SUCCESS"
      status_bg = (205, 235, 215)
      status_fg = (40, 130, 75)
      detail_text = "All boxes delivered."
      detail_colour = (60, 120, 85)

    elif truncated:
      status_text = "STEP LIMIT"
      status_bg = (248, 225, 215)
      status_fg = (185, 90, 55)
      detail_text = "Episode finished."
      detail_colour = (150, 85, 60)

    else:
      status_text = "RUNNING"
      status_bg = (197, 215, 255)
      status_fg = (55, 95, 210)
      detail_text = "Simulation in progress."
      detail_colour = (125, 130, 145)

    badge_rect = pygame.Rect(
      x + 14,
      400,
      138,
      30,
    )

    pygame.draw.rect(
      self.screen,
      status_bg,
      badge_rect,
      border_radius=15,
    )

    self._draw_centered_text(
      status_text,
      badge_rect,
      status_fg,
      self.small_font,
    )

    self._draw_text(
      detail_text,
      x + 14,
      437,
      self.small_font,
      detail_colour,
    )

    # Legend Card
    legend_card = pygame.Rect(x, 478, card_width, 130)

    pygame.draw.rect(
      self.screen,
      (250, 251, 253),
      legend_card,
      border_radius=14,
    )

    self._draw_text(
      "Legend",
      x + 14,
      492,
      self.font,
      (55, 85, 150),
    )

    legend_items = [
      ("Robot", (70, 130, 220), 0, 0),
      ("Package", (205, 145, 70), 1, 0),
      ("Finish zone", (120, 205, 130), 0, 1),
      ("Obstacle", (71, 85, 105), 1, 1),
      ("Carrying (with box)", (130, 90, 210), 0, 2),
    ]

    for label, colour, col, row in legend_items:
      item_x = x + 14 + col * 132
      item_y = 525 + row * 28

      icon_rect = pygame.Rect(item_x, item_y, 18, 18)

      pygame.draw.rect(
        self.screen,
        colour,
        icon_rect,
        border_radius=5,
      )

      pygame.draw.rect(
        self.screen,
        (55, 60, 75),
        icon_rect,
        1,
        border_radius=5,
      )

      self._draw_text(
        label,
        item_x + 26,
        item_y - 1,
        self.small_font,
        (85, 88, 96),
      )

  # Convert a warehouse grid position into a Pygame rectangle.
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

    # Render the text using the specified font and colour.
    surface = font.render(
      text,
      True,
      colour,
    )

    # Draw the rendered text at the specified position.
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

    # # Render the text and calculate its centered position.
    surface = font.render(
      text,
      True,
      colour,
    )

    text_rectangle = surface.get_rect(
      center=rectangle.center,
    )

    # Draw the text at the center of the rectangle.
    self.screen.blit(
      surface,
      text_rectangle,
    )

  def _handle_events(self) -> None:
    """Handle Pygame window events."""

    # Check for user input and window events.
    for event in pygame.event.get():

      # Stop the simulation when the window is closed.
      if event.type == pygame.QUIT:
        self.running = False

      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          self.running = False

  def _wait_for_start(self) -> None:
    """Wait for the user to click the Start button."""

    # Create the Start button.
    button = pygame.Rect(0, 0, 150, 45)
    button.center = (
      self.grid_width // 2,
      self.grid_height // 2,
    )

    # Draw a shadow.
    pygame.draw.rect(
      self.screen,
      (130, 35, 35),
      button.move(3, 4),
      border_radius=8,
    )

    # Draw the red Start button.
    pygame.draw.rect(
      self.screen,
      (220, 65, 65),
      button,
      border_radius=8,
    )

    self._draw_centered_text(
      "START",
      button,
      (255, 255, 255),
      self.font,
    )

    pygame.display.flip()

    # Wait until the Start button is clicked.
    while self.running:
      for event in pygame.event.get():

        if event.type == pygame.QUIT:
          self.running = False

        elif event.type == pygame.KEYDOWN:
          if event.key == pygame.K_ESCAPE:
            self.running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
          if event.button == 1 and button.collidepoint(event.pos):
            self.draw()
            return

      self.clock.tick(30)


  def _wait_for_next_frame(self) -> None:
    """Wait according to configured visualisation FPS."""

    # Process user events and control the frame rate.
    self._handle_events()

    self.clock.tick(self.fps)