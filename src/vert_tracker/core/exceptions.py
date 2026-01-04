"""Custom exceptions for Vert Tracker."""


class VertTrackerError(Exception):
    """Base exception for all Vert Tracker errors."""

    pass


class DroneConnectionError(VertTrackerError):
    """Failed to connect to or communicate with the Tello drone."""

    def __init__(self, message: str = "Failed to connect to drone") -> None:
        self.message = message
        super().__init__(self.message)


class VideoStreamError(VertTrackerError):
    """Error with video stream capture or processing."""

    def __init__(self, message: str = "Video stream error") -> None:
        self.message = message
        super().__init__(self.message)


class CalibrationError(VertTrackerError):
    """Calibration process failed or invalid calibration data."""

    def __init__(self, message: str = "Calibration failed") -> None:
        self.message = message
        super().__init__(self.message)


class PoseEstimationError(VertTrackerError):
    """Pose estimation failed or returned invalid data."""

    def __init__(self, message: str = "Pose estimation failed") -> None:
        self.message = message
        super().__init__(self.message)


class JumpDetectionError(VertTrackerError):
    """Jump detection algorithm encountered an error."""

    def __init__(self, message: str = "Jump detection error") -> None:
        self.message = message
        super().__init__(self.message)
