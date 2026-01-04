"""Calibration system for pixel-to-cm conversion."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from vert_tracker.core.config import CalibrationSettings
from vert_tracker.core.exceptions import CalibrationError
from vert_tracker.core.logging import get_logger
from vert_tracker.core.types import CalibrationMethod, CalibrationProfile, Frame

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

# ArUco dictionary mapping
ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
}


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
        self._detector: object | None = None
        self._current_profile: CalibrationProfile | None = None

    @property
    def current_profile(self) -> CalibrationProfile | None:
        """Get current calibration profile."""
        return self._current_profile

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration is active."""
        return self._current_profile is not None

    def _get_aruco_detector(self) -> object:
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
        corners, ids, _ = detector.detectMarkers(gray)  # type: ignore[union-attr]

        if ids is None or len(ids) == 0:
            raise CalibrationError("No ArUco marker detected in frame")

        # Use first detected marker
        marker_corners = corners[0][0]
        marker_px_size = self._calculate_marker_size(marker_corners)

        px_per_cm = marker_px_size / self.settings.aruco_marker_size_cm

        profile = CalibrationProfile(
            px_per_cm=px_per_cm,
            method=CalibrationMethod.ARUCO_MARKER,
            distance_cm=self.settings.calibration_distance_cm,
            timestamp=time.time(),
            reference_size_cm=self.settings.aruco_marker_size_cm,
        )

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
        height_normalized = abs(feet_y - head_y)
        height_px = height_normalized * frame.height

        px_per_cm = height_px / known_height_cm

        profile = CalibrationProfile(
            px_per_cm=px_per_cm,
            method=CalibrationMethod.KNOWN_HEIGHT,
            distance_cm=self.settings.calibration_distance_cm,
            timestamp=time.time(),
            reference_size_cm=known_height_cm,
        )

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

        self._current_profile = profile
        logger.info("Manual calibration: %.2f px/cm", px_per_cm)

        return profile

    def get_default_profile(self) -> CalibrationProfile:
        """Get default calibration profile from settings.

        Returns:
            Default CalibrationProfile
        """
        return CalibrationProfile(
            px_per_cm=self.settings.default_px_per_cm,
            method=CalibrationMethod.MANUAL,
            distance_cm=self.settings.calibration_distance_cm,
            timestamp=time.time(),
        )

    def _calculate_marker_size(self, corners: NDArray[np.floating]) -> float:
        """Calculate marker size in pixels from corners.

        Args:
            corners: 4x2 array of corner coordinates

        Returns:
            Average side length in pixels
        """
        side_lengths = [
            np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)
        ]
        return float(np.mean(side_lengths))

    def detect_aruco_markers(
        self, frame: Frame
    ) -> list[tuple[int, NDArray[np.floating]]]:
        """Detect all ArUco markers in frame.

        Args:
            frame: Input frame

        Returns:
            List of (marker_id, corners) tuples
        """
        detector = self._get_aruco_detector()

        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)  # type: ignore[union-attr]

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

            self._current_profile = profile
            logger.info("Loaded calibration profile from %s", path)

            return profile

        except Exception as e:
            raise CalibrationError(f"Failed to load profile: {e}") from e
