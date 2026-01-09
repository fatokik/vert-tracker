"""Core infrastructure: config, types, exceptions, and logging."""

from core.config import Settings, get_settings
from core.exceptions import (
    CalibrationError,
    DroneConnectionError,
    JumpDetectionError,
    PoseEstimationError,
    VertTrackerError,
    VideoStreamError,
)
from core.logging import get_logger, setup_logging
from core.types import (
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
