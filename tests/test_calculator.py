"""Tests for height calculation."""

from __future__ import annotations

import pytest

from vert_tracker.analysis.calculator import (
    HeightCalculator,
    calculate_jump_height,
    estimate_vertical_velocity,
)
from vert_tracker.core.types import CalibrationProfile, JumpEvent


class TestHeightCalculator:
    """Tests for the HeightCalculator class."""

    def test_calculates_positive_height(
        self,
        sample_jump_event: JumpEvent,
        calibration_profile: CalibrationProfile,
    ) -> None:
        """Should calculate positive height for upward jump."""
        calculator = HeightCalculator(calibration_profile)
        height = calculator.calculate_height(sample_jump_event, frame_height=720)

        assert height > 0

    def test_zero_displacement_gives_zero_height(
        self, calibration_profile: CalibrationProfile
    ) -> None:
        """Zero displacement should give zero height."""
        event = JumpEvent(
            takeoff_frame=0,
            peak_frame=5,
            landing_frame=10,
            height_cm=0.0,
            confidence=0.9,
            peak_hip_y=0.5,
            baseline_hip_y=0.5,  # Same as peak
        )

        calculator = HeightCalculator(calibration_profile)
        height = calculator.calculate_height(event, frame_height=720)

        assert height == 0.0

    def test_height_scales_with_calibration(self, sample_jump_event: JumpEvent) -> None:
        """Height should scale inversely with px_per_cm."""
        from vert_tracker.core.types import CalibrationMethod

        # High px_per_cm = smaller real-world height
        high_cal = CalibrationProfile(
            px_per_cm=10.0,
            method=CalibrationMethod.MANUAL,
            distance_cm=250.0,
            timestamp=0.0,
        )

        low_cal = CalibrationProfile(
            px_per_cm=2.5,
            method=CalibrationMethod.MANUAL,
            distance_cm=250.0,
            timestamp=0.0,
        )

        high_calc = HeightCalculator(high_cal)
        low_calc = HeightCalculator(low_cal)

        high_height = high_calc.calculate_height(sample_jump_event, 720)
        low_height = low_calc.calculate_height(sample_jump_event, 720)

        # Lower px_per_cm should give higher real-world height
        assert low_height > high_height
        # Ratio should match calibration ratio
        assert pytest.approx(low_height / high_height, rel=0.01) == 10.0 / 2.5

    def test_trajectory_fit_with_sufficient_points(
        self,
        calibration_profile: CalibrationProfile,
    ) -> None:
        """Trajectory fitting should work with enough points."""
        # Create event with good trajectory
        trajectory = [(i, 0.5 - 0.01 * (15 - abs(i - 15))) for i in range(30)]
        event = JumpEvent(
            takeoff_frame=0,
            peak_frame=15,
            landing_frame=30,
            height_cm=0.0,
            confidence=0.9,
            peak_hip_y=0.35,
            baseline_hip_y=0.5,
            trajectory=trajectory,
        )

        calculator = HeightCalculator(calibration_profile)
        height, fit = calculator.calculate_with_trajectory_fit(event, 720)

        assert height > 0
        assert fit.r_squared >= 0  # Fit quality metric

    def test_trajectory_fit_fallback_with_few_points(
        self,
        calibration_profile: CalibrationProfile,
    ) -> None:
        """Should fall back to simple calculation with few trajectory points."""
        event = JumpEvent(
            takeoff_frame=0,
            peak_frame=2,
            landing_frame=4,
            height_cm=0.0,
            confidence=0.9,
            peak_hip_y=0.4,
            baseline_hip_y=0.5,
            trajectory=[(0, 0.5), (2, 0.4), (4, 0.5)],  # Only 3 points
        )

        calculator = HeightCalculator(calibration_profile)
        height, fit = calculator.calculate_with_trajectory_fit(event, 720)

        assert height > 0

    def test_physics_validation(self, calibration_profile: CalibrationProfile) -> None:
        """Physics validation should check height vs airborne time."""
        calculator = HeightCalculator(calibration_profile)

        # Test with realistic values: ~40cm jump at 30fps
        # Airborne time ~0.5s means ~20cm physics height (h = 0.5 * g * (t/2)^2)
        is_valid, physics_height = calculator.validate_with_physics(
            height_cm=20.0,
            airborne_frames=15,  # 0.5 seconds
            fps=30.0,
        )

        assert physics_height > 0


class TestCalculateJumpHeight:
    """Tests for the pure function."""

    def test_pure_function_matches_class(
        self,
        sample_jump_event: JumpEvent,
        calibration_profile: CalibrationProfile,
    ) -> None:
        """Pure function should give same result as class method."""
        # Class method
        calculator = HeightCalculator(calibration_profile)
        class_height = calculator.calculate_height(sample_jump_event, 720)

        # Pure function
        func_height = calculate_jump_height(sample_jump_event, calibration_profile, 720)

        assert class_height == func_height


class TestEstimateVerticalVelocity:
    """Tests for velocity estimation."""

    def test_estimates_velocity_from_trajectory(self) -> None:
        """Should estimate velocity from trajectory points."""
        # Linear upward motion
        trajectory = [(i, 0.5 - 0.01 * i) for i in range(10)]

        velocities = estimate_vertical_velocity(trajectory, frame_height=720, fps=30.0)

        assert len(velocities) == 9  # One less than trajectory points
        # All velocities should be negative (moving up)
        assert all(v[1] < 0 for v in velocities)

    def test_empty_trajectory_returns_empty(self) -> None:
        """Empty trajectory should return empty velocities."""
        velocities = estimate_vertical_velocity([], 720, 30.0)
        assert velocities == []

    def test_single_point_returns_empty(self) -> None:
        """Single point trajectory should return empty velocities."""
        velocities = estimate_vertical_velocity([(0, 0.5)], 720, 30.0)
        assert velocities == []
