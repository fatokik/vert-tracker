"""Core data types and structures."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class Landmark:
    """A single body landmark with 3D coordinates and visibility score.

    Coordinates are normalized [0, 1] relative to frame dimensions.
    """

    x: float
    y: float
    z: float
    visibility: float

    def to_pixel(self, width: int, height: int) -> tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return int(self.x * width), int(self.y * height)


# MediaPipe pose landmark indices
class LandmarkIndex(Enum):
    """MediaPipe pose landmark indices (subset of commonly used)."""

    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass(slots=True)
class Pose:
    """Collection of body landmarks for a single frame.

    Attributes:
        landmarks: Mapping of landmark index to Landmark object
        timestamp: Frame timestamp in seconds
        frame_idx: Frame sequence number
        confidence: Overall pose detection confidence [0, 1]
    """

    landmarks: dict[int, Landmark]
    timestamp: float
    frame_idx: int
    confidence: float

    @property
    def hip_center(self) -> Landmark | None:
        """Calculate the center point between left and right hips."""
        left_hip = self.landmarks.get(LandmarkIndex.LEFT_HIP.value)
        right_hip = self.landmarks.get(LandmarkIndex.RIGHT_HIP.value)

        if left_hip is None or right_hip is None:
            return None

        return Landmark(
            x=(left_hip.x + right_hip.x) / 2,
            y=(left_hip.y + right_hip.y) / 2,
            z=(left_hip.z + right_hip.z) / 2,
            visibility=min(left_hip.visibility, right_hip.visibility),
        )

    @property
    def lowest_foot_y(self) -> float | None:
        """Get the Y coordinate of the lowest foot point (highest value = lowest position)."""
        foot_indices = [
            LandmarkIndex.LEFT_ANKLE.value,
            LandmarkIndex.RIGHT_ANKLE.value,
            LandmarkIndex.LEFT_HEEL.value,
            LandmarkIndex.RIGHT_HEEL.value,
            LandmarkIndex.LEFT_FOOT_INDEX.value,
            LandmarkIndex.RIGHT_FOOT_INDEX.value,
        ]

        y_values = [
            self.landmarks[idx].y
            for idx in foot_indices
            if idx in self.landmarks and self.landmarks[idx].visibility > 0.5
        ]

        return max(y_values) if y_values else None

    def get_landmark(self, index: LandmarkIndex) -> Landmark | None:
        """Get a specific landmark by its enum index."""
        return self.landmarks.get(index.value)


@dataclass(slots=True)
class Frame:
    """A video frame with metadata.

    Attributes:
        image: BGR image array (OpenCV format)
        timestamp: Frame timestamp in seconds
        index: Frame sequence number
        width: Frame width in pixels
        height: Frame height in pixels
    """

    image: NDArray[np.uint8]
    timestamp: float
    index: int

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.image.shape[0])

    @property
    def dimensions(self) -> tuple[int, int]:
        """Frame dimensions as (width, height)."""
        return self.width, self.height


class JumpPhase(Enum):
    """States in the jump detection state machine."""

    IDLE = auto()
    TAKEOFF = auto()
    AIRBORNE = auto()
    LANDING = auto()


@dataclass(slots=True)
class JumpEvent:
    """A detected vertical jump with calculated metrics.

    Attributes:
        takeoff_frame: Frame index when feet left ground
        peak_frame: Frame index at maximum height
        landing_frame: Frame index when feet returned to ground
        height_cm: Calculated jump height in centimeters
        confidence: Confidence score for the measurement [0, 1]
        peak_hip_y: Hip Y position at peak (normalized)
        baseline_hip_y: Hip Y position at takeoff (normalized)
        trajectory: List of (frame_idx, hip_y) points during jump
    """

    takeoff_frame: int
    peak_frame: int
    landing_frame: int
    height_cm: float
    confidence: float
    peak_hip_y: float
    baseline_hip_y: float
    trajectory: list[tuple[int, float]] = field(default_factory=list)

    @property
    def airborne_frames(self) -> int:
        """Number of frames athlete was airborne."""
        return self.landing_frame - self.takeoff_frame

    @property
    def displacement_normalized(self) -> float:
        """Hip displacement in normalized coordinates (lower y = higher jump)."""
        return self.baseline_hip_y - self.peak_hip_y


class CalibrationMethod(Enum):
    """Method used for pixel-to-cm calibration."""

    ARUCO_MARKER = auto()
    KNOWN_HEIGHT = auto()
    KNOWN_DISTANCE = auto()
    MANUAL = auto()


@dataclass(slots=True)
class CalibrationProfile:
    """Calibration data for converting pixels to real-world measurements.

    Attributes:
        px_per_cm: Pixels per centimeter at the calibration distance
        method: How the calibration was performed
        distance_cm: Distance from camera to subject in cm
        timestamp: When calibration was performed
        reference_size_cm: Size of reference object used
    """

    px_per_cm: float
    method: CalibrationMethod
    distance_cm: float
    timestamp: float
    reference_size_cm: float | None = None

    def px_to_cm(self, pixels: float) -> float:
        """Convert pixel distance to centimeters."""
        return pixels / self.px_per_cm

    def cm_to_px(self, cm: float) -> float:
        """Convert centimeters to pixel distance."""
        return cm * self.px_per_cm


@dataclass(slots=True)
class SessionStats:
    """Statistics for a training session.

    Attributes:
        jumps: List of completed jump events
        start_time: Session start timestamp
        calibration: Active calibration profile
    """

    jumps: list[JumpEvent] = field(default_factory=list)
    start_time: float = 0.0
    calibration: CalibrationProfile | None = None

    @property
    def jump_count(self) -> int:
        """Total number of valid jumps recorded."""
        return len(self.jumps)

    @property
    def max_height(self) -> float | None:
        """Maximum jump height in cm."""
        if not self.jumps:
            return None
        return max(j.height_cm for j in self.jumps)

    @property
    def avg_height(self) -> float | None:
        """Average jump height in cm."""
        if not self.jumps:
            return None
        return sum(j.height_cm for j in self.jumps) / len(self.jumps)

    @property
    def std_height(self) -> float | None:
        """Standard deviation of jump heights in cm."""
        if len(self.jumps) < 2:
            return None
        heights = [j.height_cm for j in self.jumps]
        mean = sum(heights) / len(heights)
        variance = sum((h - mean) ** 2 for h in heights) / len(heights)
        return variance**0.5

    @property
    def last_jump(self) -> JumpEvent | None:
        """Most recent jump event."""
        return self.jumps[-1] if self.jumps else None

    def add_jump(self, jump: JumpEvent) -> None:
        """Add a completed jump to the session."""
        self.jumps.append(jump)

    def reset(self) -> None:
        """Clear all recorded jumps."""
        self.jumps.clear()
