"""Pytest fixtures for Vert Tracker tests."""

from __future__ import annotations

import pytest

from vert_tracker.core.config import (
    CalibrationSettings,
    FilterSettings,
    JumpDetectionSettings,
)
from vert_tracker.core.types import (
    CalibrationMethod,
    CalibrationProfile,
    JumpEvent,
    Landmark,
    LandmarkIndex,
    Pose,
)


@pytest.fixture
def sample_landmark() -> Landmark:
    """Create a sample landmark."""
    return Landmark(x=0.5, y=0.5, z=0.0, visibility=0.95)


@pytest.fixture
def sample_pose() -> Pose:
    """Create a sample pose with basic landmarks."""
    landmarks = {
        LandmarkIndex.NOSE.value: Landmark(x=0.5, y=0.2, z=0.0, visibility=0.9),
        LandmarkIndex.LEFT_SHOULDER.value: Landmark(x=0.4, y=0.3, z=0.0, visibility=0.9),
        LandmarkIndex.RIGHT_SHOULDER.value: Landmark(x=0.6, y=0.3, z=0.0, visibility=0.9),
        LandmarkIndex.LEFT_HIP.value: Landmark(x=0.45, y=0.5, z=0.0, visibility=0.9),
        LandmarkIndex.RIGHT_HIP.value: Landmark(x=0.55, y=0.5, z=0.0, visibility=0.9),
        LandmarkIndex.LEFT_KNEE.value: Landmark(x=0.45, y=0.7, z=0.0, visibility=0.9),
        LandmarkIndex.RIGHT_KNEE.value: Landmark(x=0.55, y=0.7, z=0.0, visibility=0.9),
        LandmarkIndex.LEFT_ANKLE.value: Landmark(x=0.45, y=0.9, z=0.0, visibility=0.9),
        LandmarkIndex.RIGHT_ANKLE.value: Landmark(x=0.55, y=0.9, z=0.0, visibility=0.9),
    }
    return Pose(
        landmarks=landmarks,
        timestamp=0.0,
        frame_idx=0,
        confidence=0.9,
    )


@pytest.fixture
def standing_pose_sequence() -> list[Pose]:
    """Create a sequence of standing poses (no jump)."""
    poses = []
    for i in range(30):
        landmarks = {
            LandmarkIndex.LEFT_HIP.value: Landmark(
                x=0.45, y=0.5 + 0.001 * (i % 3 - 1), z=0.0, visibility=0.9
            ),
            LandmarkIndex.RIGHT_HIP.value: Landmark(
                x=0.55, y=0.5 + 0.001 * (i % 3 - 1), z=0.0, visibility=0.9
            ),
            LandmarkIndex.LEFT_ANKLE.value: Landmark(x=0.45, y=0.9, z=0.0, visibility=0.9),
            LandmarkIndex.RIGHT_ANKLE.value: Landmark(x=0.55, y=0.9, z=0.0, visibility=0.9),
        }
        poses.append(
            Pose(
                landmarks=landmarks,
                timestamp=i / 30.0,
                frame_idx=i,
                confidence=0.9,
            )
        )
    return poses


@pytest.fixture
def jump_pose_sequence() -> list[Pose]:
    """Create a pose sequence simulating a jump.

    Simulates: 10 frames standing, 5 frames takeoff, 15 frames airborne,
    5 frames landing, 10 frames standing.
    """
    poses = []
    frame_idx = 0

    # Standing phase (10 frames)
    baseline_y = 0.5
    for _ in range(10):
        poses.append(_create_pose_at_height(baseline_y, frame_idx))
        frame_idx += 1

    # Takeoff phase (5 frames) - moving up. Step size is large enough that
    # the resulting norm/sec velocity crosses the takeoff threshold even at
    # lower frame rates (e.g. 15 FPS), not just the 30 FPS baseline.
    takeoff_step = 0.03
    for i in range(5):
        y = baseline_y - (i + 1) * takeoff_step
        poses.append(_create_pose_at_height(y, frame_idx))
        frame_idx += 1
    takeoff_end_y = baseline_y - 5 * takeoff_step

    # Airborne phase (15 frames) - parabolic trajectory.
    # peak_y is chosen so the parabola's start (t=-1) exactly matches
    # takeoff_end_y: no spurious velocity spike/discontinuity at the
    # TAKEOFF -> AIRBORNE boundary that could be misread as a landing.
    parabola_amplitude = 0.05
    peak_y = takeoff_end_y - parabola_amplitude
    for i in range(15):
        # Parabolic motion: start at takeoff height, peak at middle, return
        t = (i - 7) / 7.0  # -1 to 1
        y = peak_y + parabola_amplitude * t * t  # Parabola
        poses.append(_create_pose_at_height(y, frame_idx))
        frame_idx += 1

    # Landing phase (5 frames) - moving down
    for i in range(5):
        y = baseline_y - 0.05 + (i + 1) * 0.01
        poses.append(_create_pose_at_height(y, frame_idx))
        frame_idx += 1

    # Standing phase (10 frames)
    for _ in range(10):
        poses.append(_create_pose_at_height(baseline_y, frame_idx))
        frame_idx += 1

    return poses


def _create_pose_at_height(hip_y: float, frame_idx: int) -> Pose:
    """Create a pose with hips at specified Y position."""
    landmarks = {
        LandmarkIndex.LEFT_HIP.value: Landmark(x=0.45, y=hip_y, z=0.0, visibility=0.9),
        LandmarkIndex.RIGHT_HIP.value: Landmark(x=0.55, y=hip_y, z=0.0, visibility=0.9),
        LandmarkIndex.LEFT_ANKLE.value: Landmark(x=0.45, y=hip_y + 0.4, z=0.0, visibility=0.9),
        LandmarkIndex.RIGHT_ANKLE.value: Landmark(x=0.55, y=hip_y + 0.4, z=0.0, visibility=0.9),
    }
    return Pose(
        landmarks=landmarks,
        timestamp=frame_idx / 30.0,
        frame_idx=frame_idx,
        confidence=0.9,
    )


@pytest.fixture
def calibration_profile() -> CalibrationProfile:
    """Create a sample calibration profile."""
    return CalibrationProfile(
        px_per_cm=5.0,
        method=CalibrationMethod.MANUAL,
        distance_cm=250.0,
        timestamp=0.0,
    )


@pytest.fixture
def sample_jump_event() -> JumpEvent:
    """Create a sample jump event."""
    return JumpEvent(
        takeoff_frame=10,
        peak_frame=22,
        landing_frame=35,
        height_cm=45.0,
        confidence=0.85,
        peak_hip_y=0.35,
        baseline_hip_y=0.5,
        trajectory=[(i, 0.5 - 0.01 * abs(22 - i)) for i in range(10, 36)],
    )


@pytest.fixture
def jump_detection_settings() -> JumpDetectionSettings:
    """Create jump detection settings for testing."""
    return JumpDetectionSettings(
        takeoff_velocity_threshold=-0.333,
        landing_velocity_threshold=0.333,
        min_airborne_s=0.167,
        max_airborne_s=2.0,
        landing_stability_frames=3,
    )


@pytest.fixture
def filter_settings() -> FilterSettings:
    """Create filter settings for testing."""
    return FilterSettings()


@pytest.fixture
def calibration_settings() -> CalibrationSettings:
    """Create calibration settings for testing."""
    return CalibrationSettings()
