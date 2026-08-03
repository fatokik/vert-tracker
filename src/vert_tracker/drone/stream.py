"""Video stream frame generator."""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from vert_tracker.core.exceptions import VideoStreamError
from vert_tracker.core.logging import get_logger
from vert_tracker.core.types import Frame

if TYPE_CHECKING:
    from vert_tracker.drone.controller import TelloController

logger = get_logger(__name__)


class VideoStream:
    """Generator-based video stream from Tello drone.

    Provides frames as Frame dataclass instances with metadata.
    """

    def __init__(self, controller: TelloController) -> None:
        """Initialize stream with drone controller.

        Args:
            controller: Connected TelloController instance
        """
        self.controller = controller
        self._frame_reader: object | None = None
        self._frame_idx = 0
        self._start_time: float | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if stream is active."""
        return self._running

    @property
    def frame_count(self) -> int:
        """Number of frames captured so far."""
        return self._frame_idx

    def start(self) -> None:
        """Initialize video streaming.

        Raises:
            VideoStreamError: If stream cannot be started
        """
        if not self.controller.is_connected:
            raise VideoStreamError("Controller not connected")

        try:
            self.controller.start_stream()
            self._frame_reader = self.controller.get_frame_reader()
            self._start_time = time.time()
            self._frame_idx = 0
            self._running = True
            logger.info("Video stream started")

        except Exception as e:
            raise VideoStreamError(f"Failed to start stream: {e}") from e

    def stop(self) -> None:
        """Stop video streaming."""
        self._running = False
        self.controller.stop_stream()
        logger.info("Video stream stopped (captured %d frames)", self._frame_idx)

    def frames(self) -> Generator[Frame, None, None]:
        """Generate Frame objects from the video stream.

        Yields:
            Frame objects with image data and metadata

        Raises:
            VideoStreamError: If stream encounters an error
        """
        if not self._running:
            self.start()

        while self._running:
            try:
                frame_data = self._read_frame()
                if frame_data is None:
                    continue

                timestamp = time.time() - (self._start_time or time.time())
                frame = Frame(
                    image=frame_data,
                    timestamp=timestamp,
                    index=self._frame_idx,
                )
                self._frame_idx += 1

                yield frame

            except StopIteration:
                break
            except Exception as e:
                logger.error("Frame capture error: %s", e)
                raise VideoStreamError(f"Frame capture failed: {e}") from e

    def _read_frame(self) -> NDArray[np.uint8] | None:
        """Read a single frame from the Tello.

        Returns:
            BGR image array or None if frame not ready
        """
        if self._frame_reader is None:
            return None

        # djitellopy's BackgroundFrameRead stores latest frame in .frame
        frame = getattr(self._frame_reader, "frame", None)
        if frame is None:
            return None

        # Ensure we have a valid numpy array
        if not isinstance(frame, np.ndarray):
            return None

        return frame.copy()

    def capture_single(self) -> Frame | None:
        """Capture a single frame without streaming.

        Returns:
            Single Frame or None if capture fails
        """
        frame_data = self._read_frame()
        if frame_data is None:
            return None

        timestamp = time.time() - (self._start_time or time.time())
        return Frame(
            image=frame_data,
            timestamp=timestamp,
            index=self._frame_idx,
        )

    def __iter__(self) -> Generator[Frame, None, None]:
        """Allow direct iteration over stream."""
        return self.frames()

    def __enter__(self) -> VideoStream:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.stop()
