"""Height calculation and trajectory analysis.

This module is pure logic with NO I/O and NO OpenCV imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.types import CalibrationProfile, JumpEvent


@dataclass(frozen=True)
class TrajectoryFit:
    """Result of fitting a parabolic trajectory."""

    peak_time: float  # Frame index of peak
    peak_height: float  # Peak height in normalized coords
    initial_velocity: float  # Estimated initial velocity
    r_squared: float  # Goodness of fit


class HeightCalculator:
    """Calculates jump height from pose displacement and calibration data.

    Supports multiple calculation methods:
    - Direct displacement: Simple pixel difference → cm conversion
    - Trajectory fitting: Fit parabola to trajectory for more accurate peak
    - Physics validation: Cross-check with expected free-fall dynamics
    """

    def __init__(self, calibration: CalibrationProfile) -> None:
        """Initialize calculator with calibration profile.

        Args:
            calibration: Active calibration for px/cm conversion
        """
        self.calibration = calibration

    def calculate_height(
        self,
        event: JumpEvent,
        frame_height: int = 720,
    ) -> float:
        """Calculate jump height in centimeters.

        Args:
            event: Jump event with trajectory data
            frame_height: Video frame height in pixels

        Returns:
            Jump height in centimeters
        """
        # Convert normalized displacement to pixels
        displacement_normalized = event.baseline_hip_y - event.peak_hip_y
        displacement_px = displacement_normalized * frame_height

        # Convert pixels to centimeters
        height_cm = self.calibration.px_to_cm(displacement_px)

        return max(0.0, height_cm)

    def calculate_with_trajectory_fit(
        self,
        event: JumpEvent,
        frame_height: int = 720,
    ) -> tuple[float, TrajectoryFit]:
        """Calculate height using parabolic trajectory fitting.

        Fits a parabola to the trajectory points to find the true peak,
        which may be between captured frames.

        Args:
            event: Jump event with trajectory data
            frame_height: Video frame height in pixels

        Returns:
            Tuple of (height_cm, trajectory_fit)
        """
        if len(event.trajectory) < 5:
            # Not enough points for fitting
            height = self.calculate_height(event, frame_height)
            fit = TrajectoryFit(
                peak_time=float(event.peak_frame),
                peak_height=event.peak_hip_y,
                initial_velocity=0.0,
                r_squared=0.0,
            )
            return height, fit

        # Extract trajectory data
        frames = np.array([t[0] for t in event.trajectory], dtype=np.float64)
        positions = np.array([t[1] for t in event.trajectory], dtype=np.float64)

        # Normalize frame indices
        frames_norm = frames - frames[0]

        # Fit parabola: y = a*t^2 + b*t + c
        try:
            fit_result = self._fit_parabola(frames_norm, positions)

            # Calculate peak from fit
            a, b, c = fit_result
            if a > 0:  # Valid upward-opening parabola (remember: y increases downward)
                peak_time = -b / (2 * a)
                peak_height = a * peak_time**2 + b * peak_time + c

                # Calculate displacement from baseline
                baseline = event.baseline_hip_y
                displacement_normalized = baseline - peak_height
                displacement_px = displacement_normalized * frame_height
                height_cm = self.calibration.px_to_cm(displacement_px)

                # Calculate R-squared
                predicted = a * frames_norm**2 + b * frames_norm + c
                ss_res = np.sum((positions - predicted) ** 2)
                ss_tot = np.sum((positions - np.mean(positions)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                trajectory_fit = TrajectoryFit(
                    peak_time=peak_time + frames[0],
                    peak_height=peak_height,
                    initial_velocity=b,
                    r_squared=r_squared,
                )

                return max(0.0, height_cm), trajectory_fit

        except Exception:
            pass

        # Fallback to simple calculation
        height = self.calculate_height(event, frame_height)
        fit = TrajectoryFit(
            peak_time=float(event.peak_frame),
            peak_height=event.peak_hip_y,
            initial_velocity=0.0,
            r_squared=0.0,
        )
        return height, fit

    def _fit_parabola(
        self,
        x: np.ndarray[tuple[int], np.dtype[np.float64]],
        y: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> tuple[float, float, float]:
        """Fit a parabola to data points.

        Args:
            x: Independent variable (frame indices)
            y: Dependent variable (positions)

        Returns:
            Coefficients (a, b, c) for y = ax^2 + bx + c
        """
        # Use numpy's polyfit for robust fitting
        coeffs = np.polyfit(x, y, 2)
        return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    def validate_with_physics(
        self,
        height_cm: float,
        airborne_frames: int,
        fps: float = 30.0,
    ) -> tuple[bool, float]:
        """Validate calculated height against physics model.

        Uses kinematic equations to check if the measured height
        is consistent with the airborne time.

        Args:
            height_cm: Calculated jump height
            airborne_frames: Number of frames airborne
            fps: Video frame rate

        Returns:
            Tuple of (is_valid, physics_predicted_height_cm)
        """
        # Airborne time in seconds
        airborne_time = airborne_frames / fps

        # Time to peak is half of airborne time (symmetric trajectory)
        time_to_peak = airborne_time / 2

        # Using h = 0.5 * g * t^2 (free fall from peak)
        g = 980.0  # cm/s^2
        physics_height = 0.5 * g * time_to_peak**2

        # Check if within reasonable tolerance (±30%)
        tolerance = 0.3
        is_valid = (
            abs(height_cm - physics_height) / physics_height < tolerance
            if physics_height > 0
            else False
        )

        return is_valid, physics_height


def calculate_jump_height(
    event: JumpEvent,
    calibration: CalibrationProfile,
    frame_height: int = 720,
) -> float:
    """Pure function to calculate jump height.

    Args:
        event: Jump event data
        calibration: Calibration profile
        frame_height: Frame height in pixels

    Returns:
        Jump height in centimeters
    """
    calculator = HeightCalculator(calibration)
    return calculator.calculate_height(event, frame_height)


def estimate_vertical_velocity(
    trajectory: list[tuple[int, float]],
    frame_height: int = 720,
    fps: float = 30.0,
) -> list[tuple[int, float]]:
    """Estimate vertical velocity from trajectory.

    Args:
        trajectory: List of (frame_idx, hip_y_normalized) points
        frame_height: Frame height in pixels
        fps: Video frame rate

    Returns:
        List of (frame_idx, velocity_cm_per_s) points
    """
    if len(trajectory) < 2:
        return []

    velocities: list[tuple[int, float]] = []

    for i in range(1, len(trajectory)):
        frame_idx = trajectory[i][0]
        dt = 1.0 / fps

        dy_normalized = trajectory[i][1] - trajectory[i - 1][1]
        dy_px = dy_normalized * frame_height
        velocity_px_per_s = dy_px / dt

        velocities.append((frame_idx, velocity_px_per_s))

    return velocities
