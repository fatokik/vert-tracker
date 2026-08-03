"""Flight lifecycle: RC stick control, keepalive, and safe shutdown."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

from vert_tracker.core.logging import get_logger
from vert_tracker.ui.display import KeyAction

logger = get_logger(__name__)

DEFAULT_RC_SPEED = 40
DEFAULT_RC_HZ = 15.0
DEFAULT_STICK_TIMEOUT_S = 0.25
DEFAULT_KEEPALIVE_S = 5.0
DEFAULT_BATTERY_CACHE_S = 1.0


class SupportsFlightControl(Protocol):
    """Minimal controller surface used by FlightSession."""

    def send_rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        ...

    def takeoff(self) -> None:
        ...

    def land(self) -> None:
        ...

    def get_battery(self) -> int:
        ...

    def stop_stream(self) -> None:
        ...

    def disconnect(self) -> None:
        ...


class FlightSession:
    """Owns airborne state, RC stick vector, keepalive, and shutdown order.

    OpenCV only reports key presses (not holds). Movement keys refresh stick
    axes; axes decay to zero after ``stick_timeout_s`` without a refresh.
    Call ``tick()`` from a background ticker or the main loop.
    """

    def __init__(
        self,
        controller: SupportsFlightControl,
        *,
        rc_speed: int = DEFAULT_RC_SPEED,
        rc_hz: float = DEFAULT_RC_HZ,
        stick_timeout_s: float = DEFAULT_STICK_TIMEOUT_S,
        keepalive_s: float = DEFAULT_KEEPALIVE_S,
        battery_cache_s: float = DEFAULT_BATTERY_CACHE_S,
        clock: Callable[[], float] = time.monotonic,
        auto_start_ticker: bool = True,
    ) -> None:
        self._controller = controller
        self._rc_speed = max(1, min(100, rc_speed))
        self._rc_interval = 1.0 / rc_hz if rc_hz > 0 else 0.05
        self._stick_timeout_s = stick_timeout_s
        self._keepalive_s = keepalive_s
        self._battery_cache_s = battery_cache_s
        self._clock = clock

        self._airborne = False
        self._lr = 0
        self._fb = 0
        self._ud = 0
        self._yaw = 0
        self._last_stick_input_at = 0.0
        self._last_rc_sent_at = 0.0
        self._last_command_at = 0.0
        self._battery: int | None = None
        self._battery_read_at = 0.0

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ticker: threading.Thread | None = None
        self._shutdown = False

        if auto_start_ticker:
            self.start_ticker()

    @property
    def is_airborne(self) -> bool:
        """Whether takeoff has succeeded and land has not completed."""
        return self._airborne

    @property
    def battery(self) -> int:
        """Cached battery percentage from the controller/state stream."""
        now = self._clock()
        if self._battery is None or (now - self._battery_read_at) >= self._battery_cache_s:
            self._battery = int(self._controller.get_battery())
            self._battery_read_at = now
        return self._battery

    def start_ticker(self) -> None:
        """Start background RC/keepalive ticker if not already running."""
        if self._ticker is not None and self._ticker.is_alive():
            return
        self._stop_event.clear()
        self._ticker = threading.Thread(
            target=self._ticker_loop,
            name="flight-session-rc",
            daemon=True,
        )
        self._ticker.start()

    def stop_ticker(self) -> None:
        """Stop background ticker and wait briefly for exit."""
        self._stop_event.set()
        if self._ticker is not None and self._ticker.is_alive():
            self._ticker.join(timeout=1.0)
        self._ticker = None

    def takeoff(self) -> None:
        """Take off and mark airborne."""
        self._controller.takeoff()
        with self._lock:
            self._airborne = True
            self._last_command_at = self._clock()
        logger.info("FlightSession: airborne")

    def land(self) -> None:
        """Land and clear airborne / sticks."""
        with self._lock:
            self._zero_sticks_unlocked()
        try:
            self._send_rc(0, 0, 0, 0)
        except Exception as e:
            logger.warning("Failed to zero RC before land: %s", e)
        self._controller.land()
        with self._lock:
            self._airborne = False
            self._last_command_at = self._clock()
        logger.info("FlightSession: landed")

    def apply_key_action(self, action: KeyAction) -> None:
        """Update stick vector from a positioning-mode key action."""
        speed = self._rc_speed
        with self._lock:
            if action == KeyAction.MOVE_FORWARD:
                self._fb = speed
            elif action == KeyAction.MOVE_BACKWARD:
                self._fb = -speed
            elif action == KeyAction.MOVE_LEFT:
                self._lr = -speed
            elif action == KeyAction.MOVE_RIGHT:
                self._lr = speed
            elif action == KeyAction.MOVE_UP:
                self._ud = speed
            elif action == KeyAction.MOVE_DOWN:
                self._ud = -speed
            elif action == KeyAction.ROTATE_LEFT:
                self._yaw = -speed
            elif action == KeyAction.ROTATE_RIGHT:
                self._yaw = speed
            else:
                return
            self._last_stick_input_at = self._clock()

    def clear_sticks(self) -> None:
        """Force all RC axes to zero."""
        with self._lock:
            self._zero_sticks_unlocked()

    def tick(self) -> None:
        """Evaluate stick timeout, send RC, and keepalive if needed."""
        if self._shutdown:
            return

        now = self._clock()
        send_zero_after_timeout = False
        with self._lock:
            airborne = self._airborne
            if (now - self._last_stick_input_at) >= self._stick_timeout_s:
                had_motion = any(v != 0 for v in (self._lr, self._fb, self._ud, self._yaw))
                if had_motion:
                    self._zero_sticks_unlocked()
                    send_zero_after_timeout = True
            lr, fb, ud, yaw = self._lr, self._fb, self._ud, self._yaw
            last_command_at = self._last_command_at

        if not airborne:
            return

        active = any(v != 0 for v in (lr, fb, ud, yaw))
        if active:
            self._send_rc(lr, fb, ud, yaw)
            return

        if send_zero_after_timeout or (now - last_command_at) >= self._keepalive_s:
            self._send_rc(0, 0, 0, 0)

    def shutdown_safe(self) -> None:
        """Stop RC, land if airborne, stop stream, disconnect."""
        if self._shutdown:
            return
        self._shutdown = True
        self.stop_ticker()

        with self._lock:
            airborne = self._airborne
            self._zero_sticks_unlocked()

        try:
            self._send_rc(0, 0, 0, 0)
        except Exception as e:
            logger.warning("Failed to zero RC during shutdown: %s", e)

        if airborne:
            try:
                self._controller.land()
                with self._lock:
                    self._airborne = False
            except Exception as e:
                logger.error("Land failed during shutdown: %s", e)

        try:
            self._controller.stop_stream()
        except Exception as e:
            logger.warning("stop_stream failed during shutdown: %s", e)

        try:
            self._controller.disconnect()
        except Exception as e:
            logger.warning("disconnect failed during shutdown: %s", e)

        logger.info("FlightSession: shutdown complete")

    def _ticker_loop(self) -> None:
        while not self._stop_event.wait(self._rc_interval):
            try:
                self.tick()
            except Exception as e:
                logger.warning("FlightSession tick error: %s", e)

    def _send_rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        self._controller.send_rc(lr, fb, ud, yaw)
        now = self._clock()
        with self._lock:
            self._last_rc_sent_at = now
            self._last_command_at = now

    def _zero_sticks_unlocked(self) -> None:
        self._lr = 0
        self._fb = 0
        self._ud = 0
        self._yaw = 0
