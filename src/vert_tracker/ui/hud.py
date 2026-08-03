"""Heads-up display (HUD) rendering for training metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from vert_tracker.core.config import UISettings
from vert_tracker.core.types import JumpEvent, SessionStats

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class HUDLayout:
    """Layout configuration for HUD elements."""

    # Metrics panel (top-left)
    metrics_x: int = 20
    metrics_y: int = 40
    metrics_line_height: int = 35

    # Phase indicator (top-right)
    phase_margin: int = 20

    # Jump history (bottom)
    history_height: int = 80
    history_bar_width: int = 40
    history_bar_gap: int = 10

    # Colors (BGR)
    color_text: tuple[int, int, int] = (255, 255, 255)
    color_bg: tuple[int, int, int] = (0, 0, 0)
    color_accent: tuple[int, int, int] = (0, 255, 255)
    color_success: tuple[int, int, int] = (0, 255, 0)
    color_warning: tuple[int, int, int] = (0, 165, 255)


class HUDRenderer:
    """Renders heads-up display elements for training feedback.

    Provides visual feedback including:
    - Current jump metrics
    - Session statistics
    - Jump history bar chart
    - Phase indicator
    - Battery/status indicators
    """

    def __init__(
        self,
        settings: UISettings | None = None,
        layout: HUDLayout | None = None,
    ) -> None:
        """Initialize HUD renderer.

        Args:
            settings: UI settings
            layout: HUD layout configuration
        """
        self.settings = settings or UISettings()
        self.layout = layout or HUDLayout()

    def render_metrics_panel(
        self,
        image: NDArray[np.uint8],
        stats: SessionStats,
        current_phase: str = "IDLE",
    ) -> NDArray[np.uint8]:
        """Render the main metrics panel.

        Args:
            image: Input image
            stats: Current session statistics
            current_phase: Current jump phase name

        Returns:
            Image with metrics panel overlay
        """
        result = image.copy()
        x = self.layout.metrics_x
        y = self.layout.metrics_y
        line_h = self.layout.metrics_line_height

        lines = [
            f"Jumps: {stats.jump_count}",
        ]

        if stats.last_jump:
            lines.append(f"Last: {stats.last_jump.height_cm:.1f} cm")

        if stats.max_height is not None:
            lines.append(f"Max: {stats.max_height:.1f} cm")

        if stats.avg_height is not None:
            lines.append(f"Avg: {stats.avg_height:.1f} cm")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        for i, text in enumerate(lines):
            pos = (x, y + i * line_h)
            self._draw_text_with_bg(result, text, pos, font, font_scale, thickness)

        return result

    def render_phase_indicator(
        self,
        image: NDArray[np.uint8],
        phase: str,
    ) -> NDArray[np.uint8]:
        """Render jump phase indicator.

        Args:
            image: Input image
            phase: Current phase name

        Returns:
            Image with phase indicator
        """
        result = image.copy()
        h, w = result.shape[:2]

        phase_colors = {
            "IDLE": (128, 128, 128),
            "TAKEOFF": (0, 255, 255),
            "AIRBORNE": (0, 255, 0),
            "LANDING": (0, 165, 255),
        }
        color = phase_colors.get(phase, self.layout.color_text)

        text = phase
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = w - text_w - self.layout.phase_margin
        y = 50

        # Background pill
        padding = 15
        cv2.rectangle(
            result,
            (x - padding, y - text_h - padding // 2),
            (x + text_w + padding, y + padding // 2),
            self.layout.color_bg,
            -1,
        )
        cv2.rectangle(
            result,
            (x - padding, y - text_h - padding // 2),
            (x + text_w + padding, y + padding // 2),
            color,
            2,
        )

        cv2.putText(result, text, (x, y), font, font_scale, color, thickness)

        return result

    def render_jump_history(
        self,
        image: NDArray[np.uint8],
        jumps: list[JumpEvent],
        max_display: int = 10,
    ) -> NDArray[np.uint8]:
        """Render jump history bar chart.

        Args:
            image: Input image
            jumps: List of jump events
            max_display: Maximum bars to display

        Returns:
            Image with history chart
        """
        if not jumps:
            return image

        result = image.copy()
        h, w = result.shape[:2]

        # Get recent jumps
        recent = jumps[-max_display:]
        max_height = max(j.height_cm for j in recent) if recent else 1.0

        bar_w = self.layout.history_bar_width
        gap = self.layout.history_bar_gap
        chart_h = self.layout.history_height
        chart_y = h - chart_h - 20

        # Draw chart background
        total_w = len(recent) * (bar_w + gap)
        chart_x = (w - total_w) // 2

        cv2.rectangle(
            result,
            (chart_x - 10, chart_y - 10),
            (chart_x + total_w + 10, h - 10),
            self.layout.color_bg,
            -1,
        )

        # Draw bars
        for i, jump in enumerate(recent):
            bar_x = chart_x + i * (bar_w + gap)
            bar_height = int((jump.height_cm / max_height) * (chart_h - 20))
            bar_y = chart_y + chart_h - bar_height - 10

            # Color based on relative height
            if jump.height_cm == max_height:
                color = self.layout.color_success
            else:
                color = self.layout.color_accent

            cv2.rectangle(
                result,
                (bar_x, bar_y),
                (bar_x + bar_w, chart_y + chart_h - 10),
                color,
                -1,
            )

            # Height label
            label = f"{jump.height_cm:.0f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (label_w, _), _ = cv2.getTextSize(label, font, 0.4, 1)
            label_x = bar_x + (bar_w - label_w) // 2
            cv2.putText(
                result,
                label,
                (label_x, bar_y - 5),
                font,
                0.4,
                self.layout.color_text,
                1,
            )

        return result

    def render_big_number(
        self,
        image: NDArray[np.uint8],
        value: float,
        label: str = "cm",
        position: tuple[int, int] | None = None,
    ) -> NDArray[np.uint8]:
        """Render a large number display (for last jump).

        Args:
            image: Input image
            value: Number to display
            label: Unit label
            position: (x, y) position or None for center

        Returns:
            Image with big number overlay
        """
        result = image.copy()
        h, w = result.shape[:2]

        text = f"{value:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4.0
        thickness = 4

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        if position is None:
            x = (w - text_w) // 2
            y = h // 2
        else:
            x, y = position

        # Semi-transparent background
        overlay = result.copy()
        padding = 40
        cv2.rectangle(
            overlay,
            (x - padding, y - text_h - padding),
            (x + text_w + padding + 80, y + padding),
            self.layout.color_bg,
            -1,
        )
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)

        # Main number
        cv2.putText(
            result,
            text,
            (x, y),
            font,
            font_scale,
            self.layout.color_success,
            thickness,
        )

        # Unit label
        cv2.putText(
            result,
            label,
            (x + text_w + 10, y),
            font,
            1.5,
            self.layout.color_text,
            2,
        )

        return result

    def render_status_bar(
        self,
        image: NDArray[np.uint8],
        battery: int | None = None,
        fps: float | None = None,
        calibrated: bool = False,
    ) -> NDArray[np.uint8]:
        """Render status bar with system info.

        Args:
            image: Input image
            battery: Drone battery percentage
            fps: Current frame rate
            calibrated: Whether system is calibrated

        Returns:
            Image with status bar
        """
        result = image.copy()
        h, w = result.shape[:2]

        items = []

        if battery is not None:
            color = self.layout.color_success if battery > 20 else self.layout.color_warning
            items.append((f"BAT: {battery}%", color))

        if fps is not None:
            items.append((f"FPS: {fps:.1f}", self.layout.color_text))

        cal_text = "CAL: OK" if calibrated else "CAL: ---"
        cal_color = self.layout.color_success if calibrated else self.layout.color_warning
        items.append((cal_text, cal_color))

        # Draw status bar
        bar_y = h - 25
        x = 20
        font = cv2.FONT_HERSHEY_SIMPLEX

        for text, color in items:
            cv2.putText(result, text, (x, bar_y), font, 0.5, color, 1)
            (text_w, _), _ = cv2.getTextSize(text, font, 0.5, 1)
            x += text_w + 30

        return result

    def _draw_text_with_bg(
        self,
        image: NDArray[np.uint8],
        text: str,
        position: tuple[int, int],
        font: int,
        font_scale: float,
        thickness: int,
    ) -> None:
        """Draw text with background rectangle.

        Args:
            image: Image to draw on (modified in place)
            text: Text to draw
            position: (x, y) position
            font: OpenCV font
            font_scale: Font scale
            thickness: Text thickness
        """
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position
        padding = 5

        cv2.rectangle(
            image,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + padding),
            self.layout.color_bg,
            -1,
        )

        cv2.putText(
            image,
            text,
            position,
            font,
            font_scale,
            self.layout.color_text,
            thickness,
        )

    def render_full_hud(
        self,
        image: NDArray[np.uint8],
        stats: SessionStats,
        phase: str = "IDLE",
        battery: int | None = None,
        fps: float | None = None,
        show_history: bool = True,
    ) -> NDArray[np.uint8]:
        """Render complete HUD overlay.

        Args:
            image: Input image
            stats: Session statistics
            phase: Current jump phase
            battery: Drone battery level
            fps: Current frame rate
            show_history: Whether to show jump history

        Returns:
            Image with full HUD
        """
        result = image.copy()

        # Metrics panel
        result = self.render_metrics_panel(result, stats, phase)

        # Phase indicator
        result = self.render_phase_indicator(result, phase)

        # Jump history
        if show_history and stats.jumps:
            result = self.render_jump_history(result, stats.jumps)

        # Status bar
        result = self.render_status_bar(
            result,
            battery=battery,
            fps=fps,
            calibrated=stats.calibration is not None,
        )

        return result
