"""Signal filtering utilities for landmark smoothing."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.config import FilterSettings
from core.logging import get_logger

logger = get_logger(__name__)


class KalmanFilter2D:
    """2D Kalman filter for smoothing (x, y) position tracking.

    State vector: [x, y, vx, vy] (position and velocity)
    Measurement: [x, y] (position only)
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ) -> None:
        """Initialize Kalman filter.

        Args:
            process_noise: Process noise covariance (higher = trust measurements more)
            measurement_noise: Measurement noise covariance (higher = trust predictions more)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # State transition matrix (constant velocity model)
        self.F: NDArray[np.floating[Any]] = np.array(
            [
                [1, 0, 1, 0],  # x = x + vx
                [0, 1, 0, 1],  # y = y + vy
                [0, 0, 1, 0],  # vx = vx
                [0, 0, 0, 1],  # vy = vy
            ],
            dtype=np.float64,
        )

        # Measurement matrix (observe position only)
        self.H: NDArray[np.floating[Any]] = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float64,
        )

        # Process noise covariance
        self.Q: NDArray[np.floating[Any]] = np.eye(4, dtype=np.float64) * process_noise

        # Measurement noise covariance
        self.R: NDArray[np.floating[Any]] = np.eye(2, dtype=np.float64) * measurement_noise

        # State estimate and covariance
        self.x: NDArray[np.floating[Any]] = np.zeros(4, dtype=np.float64)
        self.P: NDArray[np.floating[Any]] = np.eye(4, dtype=np.float64)

        self._initialized = False

    @property
    def position(self) -> tuple[float, float]:
        """Current position estimate (x, y)."""
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        """Current velocity estimate (vx, vy)."""
        return float(self.x[2]), float(self.x[3])

    def reset(self) -> None:
        """Reset filter state."""
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)
        self._initialized = False

    def predict(self) -> tuple[float, float]:
        """Predict next state.

        Returns:
            Predicted (x, y) position
        """
        # State prediction
        self.x = self.F @ self.x

        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.position

    def update(self, x: float, y: float) -> tuple[float, float]:
        """Update state with new measurement.

        Args:
            x: Measured x position
            y: Measured y position

        Returns:
            Filtered (x, y) position
        """
        measurement = np.array([x, y], dtype=np.float64)

        if not self._initialized:
            self.x[0] = x
            self.x[1] = y
            self._initialized = True
            return self.position

        # Predict first
        self.predict()

        # Measurement residual
        y_residual = measurement - self.H @ self.x

        # Residual covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y_residual

        # Covariance update
        identity = np.eye(4, dtype=np.float64)
        self.P = (identity - K @ self.H) @ self.P

        return self.position


class SmoothingFilter:
    """Moving average filter for smoothing sequences.

    Uses a sliding window to compute moving average of values.
    """

    def __init__(self, window_size: int = 5) -> None:
        """Initialize smoothing filter.

        Args:
            window_size: Number of samples in sliding window
        """
        self.window_size = window_size
        self._buffer: deque[float] = deque(maxlen=window_size)

    @property
    def is_ready(self) -> bool:
        """Check if buffer is full."""
        return len(self._buffer) >= self.window_size

    def reset(self) -> None:
        """Clear filter buffer."""
        self._buffer.clear()

    def update(self, value: float) -> float:
        """Add value and return smoothed result.

        Args:
            value: New measurement

        Returns:
            Smoothed value (moving average)
        """
        self._buffer.append(value)
        return sum(self._buffer) / len(self._buffer)

    def get_derivative(self) -> float | None:
        """Estimate derivative (rate of change) from buffer.

        Returns:
            Approximate derivative or None if insufficient samples
        """
        if len(self._buffer) < 2:
            return None

        # Simple first difference
        values = list(self._buffer)
        return values[-1] - values[-2]


class LandmarkSmoother:
    """Smoothing system for all body landmarks.

    Maintains a Kalman filter for each tracked landmark index.
    """

    def __init__(self, settings: FilterSettings | None = None) -> None:
        """Initialize landmark smoother.

        Args:
            settings: Filter settings (uses defaults if None)
        """
        self.settings = settings or FilterSettings()
        self._filters: dict[int, KalmanFilter2D] = {}

    def smooth(
        self,
        landmark_idx: int,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """Smooth a landmark position.

        Args:
            landmark_idx: Landmark index
            x: Raw x position
            y: Raw y position

        Returns:
            Smoothed (x, y) position
        """
        if landmark_idx not in self._filters:
            self._filters[landmark_idx] = KalmanFilter2D(
                process_noise=self.settings.kalman_process_noise,
                measurement_noise=self.settings.kalman_measurement_noise,
            )

        return self._filters[landmark_idx].update(x, y)

    def get_velocity(self, landmark_idx: int) -> tuple[float, float] | None:
        """Get estimated velocity for a landmark.

        Args:
            landmark_idx: Landmark index

        Returns:
            Velocity (vx, vy) or None if landmark not tracked
        """
        if landmark_idx not in self._filters:
            return None

        return self._filters[landmark_idx].velocity

    def reset(self, landmark_idx: int | None = None) -> None:
        """Reset filter state.

        Args:
            landmark_idx: Specific landmark to reset, or None for all
        """
        if landmark_idx is not None:
            if landmark_idx in self._filters:
                self._filters[landmark_idx].reset()
        else:
            self._filters.clear()
