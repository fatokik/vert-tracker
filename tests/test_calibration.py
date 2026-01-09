"""Tests for calibration system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from vert_tracker.core.config import CalibrationSettings
from vert_tracker.core.exceptions import CalibrationError
from vert_tracker.core.types import CalibrationMethod, CalibrationProfile, Frame
from vert_tracker.vision.calibration import Calibrator


class TestCalibrator:
    """Tests for the Calibrator class."""

    def test_initial_state_not_calibrated(self, calibration_settings: CalibrationSettings) -> None:
        """Calibrator should start without active calibration."""
        calibrator = Calibrator(calibration_settings)
        assert not calibrator.is_calibrated
        assert calibrator.current_profile is None

    def test_manual_calibration(self, calibration_settings: CalibrationSettings) -> None:
        """Should set calibration manually."""
        calibrator = Calibrator(calibration_settings)

        profile = calibrator.calibrate_manual(px_per_cm=6.5)

        assert calibrator.is_calibrated
        assert profile.px_per_cm == 6.5
        assert profile.method == CalibrationMethod.MANUAL
        assert calibrator.current_profile == profile

    def test_default_profile(self, calibration_settings: CalibrationSettings) -> None:
        """Should provide default calibration profile."""
        calibrator = Calibrator(calibration_settings)

        profile = calibrator.get_default_profile()

        assert profile.px_per_cm == calibration_settings.default_px_per_cm
        assert profile.method == CalibrationMethod.MANUAL

    def test_height_calibration(self, calibration_settings: CalibrationSettings) -> None:
        """Should calibrate using known height."""
        calibrator = Calibrator(calibration_settings)

        # Create mock frame
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = Frame(image=image, timestamp=0.0, index=0)

        # Person with head at y=0.1, feet at y=0.9 (80% of frame)
        # If frame is 720px and person is 180cm
        # Expected: 720 * 0.8 = 576px for 180cm = 3.2 px/cm
        profile = calibrator.calibrate_with_height(
            frame=frame,
            head_y=0.1,
            feet_y=0.9,
            known_height_cm=180.0,
        )

        assert calibrator.is_calibrated
        assert profile.method == CalibrationMethod.KNOWN_HEIGHT
        assert profile.reference_size_cm == 180.0
        assert pytest.approx(profile.px_per_cm, rel=0.01) == 3.2

    def test_save_and_load_profile(self, calibration_settings: CalibrationSettings) -> None:
        """Should save and load calibration profile."""
        calibrator = Calibrator(calibration_settings)
        calibrator.calibrate_manual(px_per_cm=7.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"

            # Save
            calibrator.save_profile(path)
            assert path.exists()

            # Verify file contents
            with open(path) as f:
                data = json.load(f)
            assert data["px_per_cm"] == 7.5
            assert data["method"] == "MANUAL"

            # Create new calibrator and load
            new_calibrator = Calibrator(calibration_settings)
            loaded = new_calibrator.load_profile(path)

            assert new_calibrator.is_calibrated
            assert loaded.px_per_cm == 7.5
            assert loaded.method == CalibrationMethod.MANUAL

    def test_save_without_calibration_raises(
        self, calibration_settings: CalibrationSettings
    ) -> None:
        """Should raise error when saving without calibration."""
        calibrator = Calibrator(calibration_settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"

            with pytest.raises(CalibrationError):
                calibrator.save_profile(path)

    def test_load_invalid_file_raises(self, calibration_settings: CalibrationSettings) -> None:
        """Should raise error when loading invalid file."""
        calibrator = Calibrator(calibration_settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.json"
            path.write_text("not valid json")

            with pytest.raises(CalibrationError):
                calibrator.load_profile(path)


class TestCalibrationProfile:
    """Tests for CalibrationProfile dataclass."""

    def test_px_to_cm_conversion(self) -> None:
        """Should convert pixels to centimeters."""
        profile = CalibrationProfile(
            px_per_cm=5.0,
            method=CalibrationMethod.MANUAL,
            distance_cm=250.0,
            timestamp=0.0,
        )

        assert profile.px_to_cm(50.0) == 10.0
        assert profile.px_to_cm(25.0) == 5.0

    def test_cm_to_px_conversion(self) -> None:
        """Should convert centimeters to pixels."""
        profile = CalibrationProfile(
            px_per_cm=5.0,
            method=CalibrationMethod.MANUAL,
            distance_cm=250.0,
            timestamp=0.0,
        )

        assert profile.cm_to_px(10.0) == 50.0
        assert profile.cm_to_px(5.0) == 25.0

    def test_conversions_are_inverse(self) -> None:
        """px_to_cm and cm_to_px should be inverse operations."""
        profile = CalibrationProfile(
            px_per_cm=4.2,
            method=CalibrationMethod.MANUAL,
            distance_cm=250.0,
            timestamp=0.0,
        )

        original = 42.0
        converted = profile.cm_to_px(profile.px_to_cm(original))
        assert pytest.approx(converted) == original
