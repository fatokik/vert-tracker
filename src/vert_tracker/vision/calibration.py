"""Calibration system for pixel-to-cm conversion."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vert_tracker.core.config import CalibrationSettings
from vert_tracker.core.exceptions import CalibrationError
from vert_tracker.core.logging import get_logger
from vert_tracker.core.types import CalibrationMethod, CalibrationProfile, Frame

logger = get_logger(__name__)

DEFAULT_CALIBRATION_PATH = Path("data/calibration/profile.json")

# ArUco dictionary mapping
ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
}


def validate_profile(profile: CalibrationProfile) -> None:
    """Validate a calibration profile before use.

    Raises:
        CalibrationError: If values are non-finite or non-positive
    """
    if not math.isfinite(profile.px_per_cm) or profile.px_per_cm <= 0:
        raise CalibrationError(
            f"px_per_cm must be a positive finite number, got {profile.px_per_cm}"
        )
    if not math.isfinite(profile.distance_cm) or profile.distance_cm <= 0:
        raise CalibrationError(
            f"distance_cm must be a positive finite number, got {profile.distance_cm}"
        )
    if profile.reference_size_cm is not None and (
        not math.isfinite(profile.reference_size_cm) or profile.reference_size_cm <= 0
    ):
        raise CalibrationError(
            f"reference_size_cm must be positive when set, got {profile.reference_size_cm}"
        )


def bootstrap_calibration(
    calibrator: Calibrator,
    path: Path | None = None,
    *,
    default_path: Path | None = None,
) -> tuple[CalibrationProfile, bool]:
    """Load calibration for session start.

    Order: explicit path → default file if present → settings default (uncalibrated).

    Returns:
        Tuple of (profile, is_calibrated)
    """
    resolved_default = default_path if default_path is not None else DEFAULT_CALIBRATION_PATH
    candidate = path
    if candidate is None and resolved_default.exists():
        candidate = resolved_default

    if candidate is None:
        profile = calibrator.get_default_profile()
        validate_profile(profile)
        logger.info(
            "Using default calibration %.2f px/cm (uncalibrated)",
            profile.px_per_cm,
        )
        return profile, False

    profile = calibrator.load_profile(candidate)
    validate_profile(profile)
    logger.info("Loaded calibration from %s (%.2f px/cm)", candidate, profile.px_per_cm)
    return profile, True


class Calibrator:
    """Calibration system for establishing pixel-to-cm ratio.

    Supports multiple calibration methods:
    - ArUco marker detection
    - Known height reference (e.g., athlete's standing height)
    - Manual specification
    """

    def __init__(self, settings: CalibrationSettings | None = None) -> None:
        """Initialize calibrator with settings.

        Args:
            settings: Calibration settings (uses defaults if None)
        """
        self.settings = settings or CalibrationSettings()
        self._detector: cv2.aruco.ArucoDetector | None = None
        self._current_profile: CalibrationProfile | None = None

    @property
    def current_profile(self) -> CalibrationProfile | None:
        """Get current calibration profile."""
        return self._current_profile

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration is active."""
        return self._current_profile is not None

    def _get_aruco_detector(self) -> cv2.aruco.ArucoDetector:
        """Get or create ArUco detector."""
        if self._detector is None:
            dict_type = ARUCO_DICTS.get(self.settings.aruco_dict, cv2.aruco.DICT_4X4_50)
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
            parameters = cv2.aruco.DetectorParameters()
            self._detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        return self._detector

    def calibrate_with_aruco(self, frame: Frame) -> CalibrationProfile:
        """Calibrate using ArUco marker in frame.

        Args:
            frame: Frame containing visible ArUco marker

        Returns:
            CalibrationProfile with computed px_per_cm

        Raises:
            CalibrationError: If marker not detected or invalid
        """
        detector = self._get_aruco_detector()

        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            raise CalibrationError("No ArUco marker detected in frame")

        if self.settings.aruco_marker_size_cm <= 0:
            raise CalibrationError("aruco_marker_size_cm must be positive")

        # Use first detected marker
        marker_corners = corners[0][0]
        marker_px_size = self._calculate_marker_size(marker_corners)
        if marker_px_size <= 0:
            raise CalibrationError("Detected ArUco marker size is invalid")

        px_per_cm = marker_px_size / self.settings.aruco_marker_size_cm

        profile = CalibrationProfile(
            px_per_cm=px_per_cm,
            method=CalibrationMethod.ARUCO_MARKER,
            distance_cm=self.settings.calibration_distance_cm,
            timestamp=time.time(),
            reference_size_cm=self.settings.aruco_marker_size_cm,
        )
        validate_profile(profile)

        self._current_profile = profile
        logger.info(
            "ArUco calibration: %.2f px/cm (marker: %d px)",
            px_per_cm,
            marker_px_size,
        )

        return profile

    def calibrate_with_height(
        self,
        frame: Frame,
        head_y: float,
        feet_y: float,
        known_height_cm: float,
    ) -> CalibrationProfile:
        """Calibrate using person's known standing height.

        Args:
            frame: Current frame
            head_y: Y coordinate of head (normalized 0-1)
            feet_y: Y coordinate of feet (normalized 0-1)
            known_height_cm: Person's actual height in cm

        Returns:
            CalibrationProfile with computed px_per_cm
        """
        if known_height_cm <= 0:
            raise CalibrationError("known_height_cm must be positive")

        height_normalized = abs(feet_y - head_y)
        height_px = height_normalized * frame.height
        if height_px <= 0:
            raise CalibrationError("Measured person height in pixels is invalid")

        px_per_cm = height_px / known_height_cm

        profile = CalibrationProfile(
            px_per_cm=px_per_cm,
            method=CalibrationMethod.KNOWN_HEIGHT,
            distance_cm=self.settings.calibration_distance_cm,
            timestamp=time.time(),
            reference_size_cm=known_height_cm,
        )
        validate_profile(profile)

        self._current_profile = profile
        logger.info(
            "Height calibration: %.2f px/cm (height: %.0f cm -> %.0f px)",
            px_per_cm,
            known_height_cm,
            height_px,
        )

        return profile

    def calibrate_manual(self, px_per_cm: float) -> CalibrationProfile:
        """Set calibration manually.

        Args:
            px_per_cm: Known pixels per centimeter value

        Returns:
            CalibrationProfile with specified value
        """
        profile = CalibrationProfile(
            px_per_cm=px_per_cm,
            method=CalibrationMethod.MANUAL,
            distance_cm=self.settings.calibration_distance_cm,
            timestamp=time.time(),
        )
        validate_profile(profile)

        self._current_profile = profile
        logger.info("Manual calibration: %.2f px/cm", px_per_cm)

        return profile

    def get_default_profile(self) -> CalibrationProfile:
        """Get default calibration profile from settings.

        Returns:
            Default CalibrationProfile
        """
        profile = CalibrationProfile(
            px_per_cm=self.settings.default_px_per_cm,
            method=CalibrationMethod.MANUAL,
            distance_cm=self.settings.calibration_distance_cm,
            timestamp=time.time(),
        )
        validate_profile(profile)
        return profile

    def _calculate_marker_size(self, corners: NDArray[np.floating[Any]]) -> float:
        """Calculate marker size in pixels from corners.

        Args:
            corners: 4x2 array of corner coordinates

        Returns:
            Average side length in pixels
        """
        side_lengths = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
        return float(np.mean(side_lengths))

    def detect_aruco_markers(self, frame: Frame) -> list[tuple[int, NDArray[np.floating[Any]]]]:
        """Detect all ArUco markers in frame.

        Args:
            frame: Input frame

        Returns:
            List of (marker_id, corners) tuples
        """
        detector = self._get_aruco_detector()

        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            return []

        return [(int(ids[i][0]), corners[i][0]) for i in range(len(ids))]

    def save_profile(self, path: Path) -> None:
        """Save current calibration profile to file.

        Args:
            path: Output file path (JSON)

        Raises:
            CalibrationError: If no active profile
        """
        if self._current_profile is None:
            raise CalibrationError("No calibration profile to save")

        data = {
            "px_per_cm": self._current_profile.px_per_cm,
            "method": self._current_profile.method.name,
            "distance_cm": self._current_profile.distance_cm,
            "timestamp": self._current_profile.timestamp,
            "reference_size_cm": self._current_profile.reference_size_cm,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved calibration profile to %s", path)

    def load_profile(self, path: Path) -> CalibrationProfile:
        """Load calibration profile from file.

        Args:
            path: Input file path (JSON)

        Returns:
            Loaded CalibrationProfile

        Raises:
            CalibrationError: If file invalid or not found
        """
        try:
            with open(path) as f:
                data = json.load(f)

            profile = CalibrationProfile(
                px_per_cm=data["px_per_cm"],
                method=CalibrationMethod[data["method"]],
                distance_cm=data["distance_cm"],
                timestamp=data["timestamp"],
                reference_size_cm=data.get("reference_size_cm"),
            )
            validate_profile(profile)

            self._current_profile = profile
            logger.info("Loaded calibration profile from %s", path)

            return profile

        except CalibrationError:
            raise
        except Exception as e:
            raise CalibrationError(f"Failed to load profile: {e}") from e
