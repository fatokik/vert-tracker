"""Tests for session metrics tracking."""

from __future__ import annotations

import pytest

from vert_tracker.analysis.metrics import MetricsTracker
from vert_tracker.core.types import JumpEvent


def test_airborne_time_from_timestamps() -> None:
    """add_jump should derive airborne time from JumpEvent timestamps, not a fixed fps."""
    event = JumpEvent(
        takeoff_frame=10,
        peak_frame=20,
        landing_frame=40,
        height_cm=40.0,
        confidence=0.9,
        peak_hip_y=0.35,
        baseline_hip_y=0.5,
        takeoff_timestamp=1.0,
        peak_timestamp=1.3,
        landing_timestamp=2.0,
    )
    metrics = MetricsTracker().add_jump(event)

    assert metrics.airborne_time_s == pytest.approx(1.0)


def test_airborne_time_independent_of_frame_count() -> None:
    """Two events with identical timestamps but different frame counts should agree."""
    event_dense = JumpEvent(
        takeoff_frame=0,
        peak_frame=30,
        landing_frame=60,
        height_cm=40.0,
        confidence=0.9,
        peak_hip_y=0.35,
        baseline_hip_y=0.5,
        takeoff_timestamp=0.0,
        peak_timestamp=0.5,
        landing_timestamp=1.0,
    )
    event_sparse = JumpEvent(
        takeoff_frame=0,
        peak_frame=5,
        landing_frame=10,
        height_cm=40.0,
        confidence=0.9,
        peak_hip_y=0.35,
        baseline_hip_y=0.5,
        takeoff_timestamp=0.0,
        peak_timestamp=0.5,
        landing_timestamp=1.0,
    )

    dense_metrics = MetricsTracker().add_jump(event_dense)
    sparse_metrics = MetricsTracker().add_jump(event_sparse)

    assert dense_metrics.airborne_time_s == pytest.approx(sparse_metrics.airborne_time_s)


def test_peak_velocity_estimate_positive_for_positive_height() -> None:
    """Peak velocity estimate should be positive whenever a positive height was recorded."""
    event = JumpEvent(
        takeoff_frame=0,
        peak_frame=10,
        landing_frame=20,
        height_cm=30.0,
        confidence=0.9,
        peak_hip_y=0.35,
        baseline_hip_y=0.5,
        takeoff_timestamp=0.0,
        peak_timestamp=0.33,
        landing_timestamp=0.67,
    )

    metrics = MetricsTracker().add_jump(event)

    assert metrics.peak_velocity_estimate > 0
    assert metrics.height_cm == 30.0
    assert metrics.confidence == 0.9
