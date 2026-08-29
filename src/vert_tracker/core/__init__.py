"""Core infrastructure: config, types, exceptions, and logging."""

from vert_tracker.core.config import Settings, get_settings
from vert_tracker.core.exceptions import (
    CalibrationError,
    DroneConnectionError,
    JumpDetectionError,
    PoseEstimationError,
    VertTrackerError,
    VideoStreamError,
)
from vert_tracker.core.logging import get_logger, setup_logging
from vert_tracker.core.types import (
    CalibrationProfile,
    Frame,
    JumpEvent,
    JumpPhase,
    Landmark,
    Pose,
    SessionStats,
)

__all__ = [
    # Config
    "Settings",
    "get_settings",
    # Types
    "Landmark",
    "Pose",
    "Frame",
    "JumpPhase",
    "JumpEvent",
    "CalibrationProfile",
    "SessionStats",
    # Exceptions
    "VertTrackerError",
    "DroneConnectionError",
    "VideoStreamError",
    "CalibrationError",
    "PoseEstimationError",
    "JumpDetectionError",
    # Logging
    "setup_logging",
    "get_logger",
]
