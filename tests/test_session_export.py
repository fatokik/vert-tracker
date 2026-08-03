"""Tests for session save wiring (main.save_session_stats)."""

from __future__ import annotations

import json
from pathlib import Path

from vert_tracker.core.types import CalibrationMethod, CalibrationProfile, JumpEvent, SessionStats
from vert_tracker.main import save_session_stats


def _make_stats() -> SessionStats:
    stats = SessionStats(start_time=0.0)
    stats.calibration = CalibrationProfile(
        px_per_cm=5.0,
        method=CalibrationMethod.MANUAL,
        distance_cm=250.0,
        timestamp=0.0,
    )
    stats.add_jump(
        JumpEvent(
            takeoff_frame=0,
            peak_frame=5,
            landing_frame=10,
            height_cm=42.0,
            confidence=0.9,
            peak_hip_y=0.2,
            baseline_hip_y=0.5,
            takeoff_timestamp=0.0,
            peak_timestamp=0.17,
            landing_timestamp=0.34,
        )
    )
    return stats


def test_save_session_stats_writes_file_under_directory(tmp_path: Path) -> None:
    stats = _make_stats()

    result_path = save_session_stats(stats, directory=tmp_path)

    assert result_path.exists()
    assert result_path.parent == tmp_path

    with open(result_path) as f:
        data = json.load(f)
    assert data["jump_count"] == 1
    assert data["jumps"][0]["height_cm"] == 42.0


def test_save_session_stats_defaults_to_data_sessions_directory() -> None:
    stats = _make_stats()

    result_path = save_session_stats(stats)

    assert result_path.parent == Path("data/sessions")
    assert result_path.exists()

    result_path.unlink()
