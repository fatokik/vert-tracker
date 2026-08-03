"""Tests for calibration bootstrap and model path resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vert_tracker.core.exceptions import CalibrationError
from vert_tracker.core.types import CalibrationMethod
from vert_tracker.vision.calibration import (
    DEFAULT_CALIBRATION_PATH,
    Calibrator,
    bootstrap_calibration,
    validate_profile,
)
from vert_tracker.vision.pose import default_model_dir, resolve_project_root


def test_validate_profile_rejects_non_positive_px_per_cm() -> None:
    from vert_tracker.core.types import CalibrationProfile

    profile = CalibrationProfile(
        px_per_cm=0.0,
        method=CalibrationMethod.MANUAL,
        distance_cm=250.0,
        timestamp=0.0,
    )
    with pytest.raises(CalibrationError):
        validate_profile(profile)


def test_manual_calibration_rejects_negative() -> None:
    calibrator = Calibrator()
    with pytest.raises(CalibrationError):
        calibrator.calibrate_manual(-1.0)


def test_bootstrap_missing_uses_default_uncalibrated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(
        "vert_tracker.vision.calibration.DEFAULT_CALIBRATION_PATH",
        missing,
    )
    profile, calibrated = bootstrap_calibration(Calibrator(), path=None)
    assert not calibrated
    assert profile.px_per_cm > 0
    assert profile.method == CalibrationMethod.MANUAL


def test_bootstrap_loads_valid_profile(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "px_per_cm": 4.5,
                "method": "MANUAL",
                "distance_cm": 250.0,
                "timestamp": 1.0,
                "reference_size_cm": None,
            }
        )
    )
    profile, calibrated = bootstrap_calibration(Calibrator(), path=path)
    assert calibrated
    assert profile.px_per_cm == 4.5


def test_bootstrap_rejects_invalid_saved_profile(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "px_per_cm": 0,
                "method": "MANUAL",
                "distance_cm": 250.0,
                "timestamp": 1.0,
            }
        )
    )
    with pytest.raises(CalibrationError):
        bootstrap_calibration(Calibrator(), path=path)


def test_model_dir_is_under_project_data() -> None:
    root = resolve_project_root()
    model_dir = default_model_dir()
    assert model_dir == root / "data" / "models"
    assert (root / "pyproject.toml").exists()


def test_default_calibration_path_constant() -> None:
    assert Path("data/calibration/profile.json") == DEFAULT_CALIBRATION_PATH
