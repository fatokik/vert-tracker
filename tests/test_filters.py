"""Tests for signal filtering utilities."""

from __future__ import annotations

import pytest
from vert_tracker.core.config import FilterSettings
from vert_tracker.vision.filters import KalmanFilter2D, LandmarkSmoother, SmoothingFilter


class TestKalmanFilter2D:
    """Tests for the Kalman filter."""

    def test_initial_update_returns_input(self) -> None:
        """First update should return the input value."""
        kf = KalmanFilter2D()
        x, y = kf.update(0.5, 0.5)

        assert x == 0.5
        assert y == 0.5

    def test_filters_noisy_signal(self) -> None:
        """Should smooth out noise in measurements."""
        kf = KalmanFilter2D(process_noise=0.01, measurement_noise=0.5)

        # Noisy measurements around (0.5, 0.5)
        import random

        random.seed(42)

        filtered_positions = []
        for _ in range(50):
            noisy_x = 0.5 + random.gauss(0, 0.05)
            noisy_y = 0.5 + random.gauss(0, 0.05)
            x, y = kf.update(noisy_x, noisy_y)
            filtered_positions.append((x, y))

        # Later positions should be closer to true value
        late_x = [p[0] for p in filtered_positions[-10:]]
        late_y = [p[1] for p in filtered_positions[-10:]]

        # Mean of late positions should be close to 0.5
        assert abs(sum(late_x) / len(late_x) - 0.5) < 0.05
        assert abs(sum(late_y) / len(late_y) - 0.5) < 0.05

    def test_tracks_moving_target(self) -> None:
        """Should track a smoothly moving target."""
        kf = KalmanFilter2D(process_noise=0.1, measurement_noise=0.01)

        # Linear motion
        for i in range(20):
            true_x = 0.1 + i * 0.02
            true_y = 0.5
            x, y = kf.update(true_x, true_y)

        # Should be close to final position
        assert abs(x - 0.48) < 0.1
        assert abs(y - 0.5) < 0.05

    def test_predict_extrapolates(self) -> None:
        """Predict should extrapolate based on velocity."""
        kf = KalmanFilter2D()

        # Establish velocity with two updates
        kf.update(0.0, 0.5)
        kf.update(0.1, 0.5)

        # Predict next position
        pred_x, pred_y = kf.predict()

        # Should predict forward motion
        assert pred_x > 0.1

    def test_reset_clears_state(self) -> None:
        """Reset should clear filter state."""
        kf = KalmanFilter2D()

        # Establish state
        kf.update(0.5, 0.5)
        kf.update(0.6, 0.5)

        kf.reset()

        # Next update should be at input position
        x, y = kf.update(0.0, 0.0)
        assert x == 0.0
        assert y == 0.0

    def test_velocity_property(self) -> None:
        """Should estimate velocity from motion."""
        kf = KalmanFilter2D()

        # Moving right
        kf.update(0.0, 0.5)
        kf.update(0.1, 0.5)
        kf.update(0.2, 0.5)

        vx, vy = kf.velocity

        # Should have positive x velocity
        assert vx > 0
        # Should have near-zero y velocity
        assert abs(vy) < 0.01


class TestSmoothingFilter:
    """Tests for the moving average filter."""

    def test_smooths_values(self) -> None:
        """Should compute moving average."""
        sf = SmoothingFilter(window_size=3)

        result1 = sf.update(1.0)
        result2 = sf.update(2.0)
        result3 = sf.update(3.0)

        assert result1 == 1.0  # Only one value
        assert result2 == 1.5  # Average of 1, 2
        assert result3 == 2.0  # Average of 1, 2, 3

    def test_sliding_window(self) -> None:
        """Should maintain sliding window."""
        sf = SmoothingFilter(window_size=3)

        sf.update(1.0)
        sf.update(2.0)
        sf.update(3.0)
        result = sf.update(6.0)  # Window is now [2, 3, 6]

        assert pytest.approx(result) == (2 + 3 + 6) / 3

    def test_is_ready_property(self) -> None:
        """is_ready should indicate when buffer is full."""
        sf = SmoothingFilter(window_size=3)

        assert not sf.is_ready
        sf.update(1.0)
        assert not sf.is_ready
        sf.update(2.0)
        assert not sf.is_ready
        sf.update(3.0)
        assert sf.is_ready

    def test_get_derivative(self) -> None:
        """Should estimate derivative from buffer."""
        sf = SmoothingFilter(window_size=5)

        # No derivative with single point
        sf.update(1.0)
        assert sf.get_derivative() is None

        # Should compute difference
        sf.update(3.0)
        deriv = sf.get_derivative()
        assert deriv == 2.0  # 3 - 1

    def test_reset_clears_buffer(self) -> None:
        """Reset should clear the buffer."""
        sf = SmoothingFilter(window_size=3)

        sf.update(1.0)
        sf.update(2.0)
        sf.update(3.0)

        sf.reset()

        assert not sf.is_ready
        result = sf.update(10.0)
        assert result == 10.0  # Only value in buffer


class TestLandmarkSmoother:
    """Tests for the landmark smoothing system."""

    def test_smooths_individual_landmarks(self, filter_settings: FilterSettings) -> None:
        """Should maintain separate filter per landmark."""
        smoother = LandmarkSmoother(filter_settings)

        # Smooth landmark 0
        x0, y0 = smoother.smooth(0, 0.5, 0.5)
        assert x0 == 0.5
        assert y0 == 0.5

        # Smooth landmark 1 (different position)
        x1, y1 = smoother.smooth(1, 0.2, 0.8)
        assert x1 == 0.2
        assert y1 == 0.8

        # Update landmark 0 again
        x0_2, y0_2 = smoother.smooth(0, 0.5, 0.5)
        # Should be smoothed, not reset
        assert (x0_2, y0_2) != (0.2, 0.8)

    def test_get_velocity_for_tracked_landmark(self, filter_settings: FilterSettings) -> None:
        """Should return velocity for tracked landmarks."""
        smoother = LandmarkSmoother(filter_settings)

        # Track landmark with motion
        smoother.smooth(0, 0.0, 0.5)
        smoother.smooth(0, 0.1, 0.5)
        smoother.smooth(0, 0.2, 0.5)

        velocity = smoother.get_velocity(0)
        assert velocity is not None
        vx, vy = velocity
        assert vx > 0  # Moving right

    def test_get_velocity_for_untracked_landmark(self, filter_settings: FilterSettings) -> None:
        """Should return None for untracked landmarks."""
        smoother = LandmarkSmoother(filter_settings)

        velocity = smoother.get_velocity(99)
        assert velocity is None

    def test_reset_specific_landmark(self, filter_settings: FilterSettings) -> None:
        """Should reset specific landmark filter."""
        smoother = LandmarkSmoother(filter_settings)

        smoother.smooth(0, 0.5, 0.5)
        smoother.smooth(1, 0.3, 0.3)

        smoother.reset(0)

        # Landmark 0 should be reset
        x0, y0 = smoother.smooth(0, 0.9, 0.9)
        assert x0 == 0.9  # First value after reset

        # Landmark 1 should still be tracked
        velocity = smoother.get_velocity(1)
        assert velocity is not None

    def test_reset_all_landmarks(self, filter_settings: FilterSettings) -> None:
        """Should reset all landmark filters."""
        smoother = LandmarkSmoother(filter_settings)

        smoother.smooth(0, 0.5, 0.5)
        smoother.smooth(1, 0.3, 0.3)

        smoother.reset()

        assert smoother.get_velocity(0) is None
        assert smoother.get_velocity(1) is None
