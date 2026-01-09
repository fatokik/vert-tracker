"""OpenCV window management for video display."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import cv2
import numpy as np

from core.config import UISettings
from core.logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


class KeyAction(Enum):
    """Actions triggered by keyboard input."""

    NONE = auto()
    QUIT = auto()
    CALIBRATE = auto()
    RESET = auto()
    SAVE = auto()
    PAUSE = auto()
    TOGGLE_SKELETON = auto()
    TOGGLE_METRICS = auto()
    TOGGLE_DEBUG = auto()


# Key mappings (ASCII codes)
KEY_BINDINGS: dict[int, KeyAction] = {
    ord("q"): KeyAction.QUIT,
    ord("Q"): KeyAction.QUIT,
    27: KeyAction.QUIT,  # ESC
    ord("c"): KeyAction.CALIBRATE,
    ord("C"): KeyAction.CALIBRATE,
    ord("r"): KeyAction.RESET,
    ord("R"): KeyAction.RESET,
    ord("s"): KeyAction.SAVE,
    ord("S"): KeyAction.SAVE,
    ord(" "): KeyAction.PAUSE,
    ord("k"): KeyAction.TOGGLE_SKELETON,
    ord("m"): KeyAction.TOGGLE_METRICS,
    ord("d"): KeyAction.TOGGLE_DEBUG,
}


@dataclass
class WindowState:
    """Current state of the display window."""

    is_open: bool = False
    is_paused: bool = False
    width: int = 1280
    height: int = 720


class DisplayWindow:
    """Manages OpenCV display window for video output.

    Handles window creation, frame display, and keyboard input.
    """

    WINDOW_NAME = "Vert Tracker"

    def __init__(self, settings: UISettings | None = None) -> None:
        """Initialize display window.

        Args:
            settings: UI settings (uses defaults if None)
        """
        self.settings = settings or UISettings()
        self._state = WindowState(
            width=self.settings.display_width,
            height=self.settings.display_height,
        )

    @property
    def is_open(self) -> bool:
        """Check if window is open."""
        return self._state.is_open

    @property
    def is_paused(self) -> bool:
        """Check if display is paused."""
        return self._state.is_paused

    def open(self) -> None:
        """Create and show the display window."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self._state.width, self._state.height)
        self._state.is_open = True
        logger.info(
            "Display window opened (%dx%d)",
            self._state.width,
            self._state.height,
        )

    def close(self) -> None:
        """Close and destroy the display window."""
        cv2.destroyWindow(self.WINDOW_NAME)
        self._state.is_open = False
        logger.info("Display window closed")

    def show_frame(self, image: NDArray[np.uint8]) -> None:
        """Display a frame in the window.

        Args:
            image: BGR image array to display
        """
        if not self._state.is_open:
            self.open()

        # Resize if needed
        h, w = image.shape[:2]
        if w != self._state.width or h != self._state.height:
            resized = cv2.resize(image, (self._state.width, self._state.height))
            cv2.imshow(self.WINDOW_NAME, resized)
        else:
            cv2.imshow(self.WINDOW_NAME, image)

    def poll_key(self, wait_ms: int = 1) -> KeyAction:
        """Poll for keyboard input.

        Args:
            wait_ms: Milliseconds to wait for key (1 for non-blocking)

        Returns:
            KeyAction corresponding to pressed key
        """
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == 255:  # No key pressed
            return KeyAction.NONE

        action = KEY_BINDINGS.get(key, KeyAction.NONE)

        # Handle pause toggle
        if action == KeyAction.PAUSE:
            self._state.is_paused = not self._state.is_paused
            logger.info("Paused: %s", self._state.is_paused)

        return action

    def wait_for_key(self) -> KeyAction:
        """Block until a key is pressed.

        Returns:
            KeyAction corresponding to pressed key
        """
        while True:
            action = self.poll_key(wait_ms=100)
            if action != KeyAction.NONE:
                return action

    def show_message(
        self,
        message: str,
        duration_ms: int = 2000,
        background: NDArray[np.uint8] | None = None,
    ) -> None:
        """Display a message overlay.

        Args:
            message: Text message to display
            duration_ms: How long to show (0 = until key press)
            background: Background image (black if None)
        """
        if background is None:
            image = np.zeros(
                (self._state.height, self._state.width, 3),
                dtype=np.uint8,
            )
        else:
            image = background.copy()

        # Draw message centered
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 2

        (text_w, text_h), _ = cv2.getTextSize(message, font, font_scale, thickness)
        x = (self._state.width - text_w) // 2
        y = (self._state.height + text_h) // 2

        # Background rectangle
        padding = 20
        cv2.rectangle(
            image,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + padding),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            image,
            message,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        self.show_frame(image)

        if duration_ms > 0:
            cv2.waitKey(duration_ms)
        else:
            self.wait_for_key()

    def __enter__(self) -> DisplayWindow:
        """Context manager entry."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.close()
