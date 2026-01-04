"""Tello drone controller wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from djitellopy import Tello

from vert_tracker.core.config import DroneSettings
from vert_tracker.core.exceptions import DroneConnectionError
from vert_tracker.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class TelloController:
    """High-level wrapper for Tello drone control.

    Provides connection management, basic flight commands, and
    integration with the video streaming system.
    """

    def __init__(self, settings: DroneSettings | None = None) -> None:
        """Initialize controller with settings.

        Args:
            settings: Drone connection settings (uses defaults if None)
        """
        self.settings = settings or DroneSettings()
        self._tello: Tello | None = None
        self._connected = False
        self._streaming = False

    @property
    def is_connected(self) -> bool:
        """Check if connected to drone."""
        return self._connected

    @property
    def is_streaming(self) -> bool:
        """Check if video stream is active."""
        return self._streaming

    @property
    def tello(self) -> Tello:
        """Get underlying Tello instance (must be connected)."""
        if self._tello is None:
            raise DroneConnectionError("Not connected to drone")
        return self._tello

    def connect(self) -> None:
        """Establish connection to Tello drone.

        Raises:
            DroneConnectionError: If connection fails
        """
        logger.info("Connecting to Tello at %s...", self.settings.ip)

        try:
            self._tello = Tello(self.settings.ip)
            self._tello.RESPONSE_TIMEOUT = int(self.settings.connect_timeout)
            self._tello.connect()
            self._connected = True

            battery = self._tello.get_battery()
            logger.info("Connected to Tello (battery: %d%%)", battery)

        except Exception as e:
            self._connected = False
            raise DroneConnectionError(f"Failed to connect: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from drone and cleanup resources."""
        if self._streaming:
            self.stop_stream()

        if self._tello is not None:
            try:
                self._tello.end()
            except Exception as e:
                logger.warning("Error during disconnect: %s", e)
            finally:
                self._tello = None
                self._connected = False

        logger.info("Disconnected from Tello")

    def start_stream(self) -> None:
        """Start video streaming from drone.

        Raises:
            DroneConnectionError: If not connected
        """
        if not self._connected:
            raise DroneConnectionError("Must connect before starting stream")

        logger.info("Starting video stream...")
        self.tello.streamon()
        self._streaming = True

    def stop_stream(self) -> None:
        """Stop video streaming."""
        if self._tello is not None and self._streaming:
            try:
                self._tello.streamoff()
            except Exception as e:
                logger.warning("Error stopping stream: %s", e)
            finally:
                self._streaming = False

    def takeoff(self) -> None:
        """Execute takeoff sequence.

        Raises:
            DroneConnectionError: If not connected
        """
        if not self._connected:
            raise DroneConnectionError("Must connect before takeoff")

        logger.info("Taking off...")
        self.tello.takeoff()

    def land(self) -> None:
        """Execute landing sequence."""
        if self._tello is not None and self._connected:
            logger.info("Landing...")
            self.tello.land()

    def hover_at_height(self, height_cm: int | None = None) -> None:
        """Move to specified hover height.

        Args:
            height_cm: Target height in cm (uses settings default if None)
        """
        target = height_cm or self.settings.hover_height_cm
        current = self.tello.get_height()

        delta = target - current
        if abs(delta) > 20:  # Only move if difference is significant
            if delta > 0:
                self.tello.move_up(min(delta, 100))
            else:
                self.tello.move_down(min(abs(delta), 100))

    def get_battery(self) -> int:
        """Get current battery level percentage."""
        return self.tello.get_battery()

    def get_height(self) -> int:
        """Get current height in cm."""
        return self.tello.get_height()

    def get_frame_reader(self) -> object:
        """Get the frame reader object for video streaming.

        Returns:
            BackgroundFrameRead instance from djitellopy
        """
        return self.tello.get_frame_read()

    def __enter__(self) -> TelloController:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit with cleanup."""
        self.disconnect()
