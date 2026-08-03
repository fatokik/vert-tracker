"""Computer vision operations: pose estimation, calibration, filtering, and overlay."""

from vert_tracker.vision.calibration import (
    DEFAULT_CALIBRATION_PATH,
    Calibrator,
    bootstrap_calibration,
    validate_profile,
)
from vert_tracker.vision.filters import KalmanFilter2D, SmoothingFilter
from vert_tracker.vision.overlay import OverlayRenderer
from vert_tracker.vision.pose import PoseEstimator, default_model_dir, resolve_project_root

__all__ = [
    "PoseEstimator",
    "Calibrator",
    "KalmanFilter2D",
    "SmoothingFilter",
    "OverlayRenderer",
    "DEFAULT_CALIBRATION_PATH",
    "bootstrap_calibration",
    "validate_profile",
    "default_model_dir",
    "resolve_project_root",
]
