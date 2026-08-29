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

MIN_MOVE_CM = 20
MAX_MOVE_CM = 500
MIN_ROTATE_DEG = 1
MAX_ROTATE_DEG = 360


def clamp_distance_cm(distance_cm: int) -> int:
    """Clamp movement distance to Tello SDK 2.0 limits (20-500 cm)."""
    return max(MIN_MOVE_CM, min(MAX_MOVE_CM, int(distance_cm)))


def clamp_degrees(degrees: int) -> int:
    """Clamp rotation angle to Tello SDK 2.0 limits (1-360 deg)."""
    return max(MIN_ROTATE_DEG, min(MAX_ROTATE_DEG, int(degrees)))


def clamp_rc(value: int) -> int:
    """Clamp an RC stick axis to -100..100."""
    return max(-100, min(100, int(value)))


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

    def send_rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        """Send non-blocking RC stick velocities (-100..100 per axis)."""
        self.tello.send_rc_control(
            clamp_rc(lr),
            clamp_rc(fb),
            clamp_rc(ud),
            clamp_rc(yaw),
        )

    def move_forward(self, distance_cm: int) -> None:
        """Move drone forward.

        Args:
            distance_cm: Distance to move (20-500 cm)
        """
        distance_cm = clamp_distance_cm(distance_cm)
        logger.debug("Moving forward %d cm", distance_cm)
        self.tello.move_forward(distance_cm)

    def move_backward(self, distance_cm: int) -> None:
        """Move drone backward.

        Args:
            distance_cm: Distance to move (20-500 cm)
        """
        distance_cm = clamp_distance_cm(distance_cm)
        logger.debug("Moving backward %d cm", distance_cm)
        self.tello.move_back(distance_cm)

    def move_left(self, distance_cm: int) -> None:
        """Move drone left.

        Args:
            distance_cm: Distance to move (20-500 cm)
        """
        distance_cm = clamp_distance_cm(distance_cm)
        logger.debug("Moving left %d cm", distance_cm)
        self.tello.move_left(distance_cm)

    def move_right(self, distance_cm: int) -> None:
        """Move drone right.

        Args:
            distance_cm: Distance to move (20-500 cm)
        """
        distance_cm = clamp_distance_cm(distance_cm)
        logger.debug("Moving right %d cm", distance_cm)
        self.tello.move_right(distance_cm)

    def move_up(self, distance_cm: int) -> None:
        """Move drone up.

        Args:
            distance_cm: Distance to move (20-500 cm)
        """
        distance_cm = clamp_distance_cm(distance_cm)
        logger.debug("Moving up %d cm", distance_cm)
        self.tello.move_up(distance_cm)

    def move_down(self, distance_cm: int) -> None:
        """Move drone down.

        Args:
            distance_cm: Distance to move (20-500 cm)
        """
        distance_cm = clamp_distance_cm(distance_cm)
        logger.debug("Moving down %d cm", distance_cm)
        self.tello.move_down(distance_cm)

    def rotate_clockwise(self, degrees: int) -> None:
        """Rotate drone clockwise.

        Args:
            degrees: Rotation angle (1-360 degrees)
        """
        degrees = clamp_degrees(degrees)
        logger.debug("Rotating clockwise %d degrees", degrees)
        self.tello.rotate_clockwise(degrees)

    def rotate_counter_clockwise(self, degrees: int) -> None:
        """Rotate drone counter-clockwise.

        Args:
            degrees: Rotation angle (1-360 degrees)
        """
        degrees = clamp_degrees(degrees)
        logger.debug("Rotating counter-clockwise %d degrees", degrees)
        self.tello.rotate_counter_clockwise(degrees)

    def get_battery(self) -> int:
        """Get current battery percentage from Tello state stream."""
        return int(self.tello.get_battery())

    def get_height(self) -> int:
        """Get current height in cm."""
        return int(self.tello.get_height())

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
