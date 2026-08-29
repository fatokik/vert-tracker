"""MediaPipe pose estimation wrapper using the Tasks API."""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from vert_tracker.core.config import PoseSettings
from vert_tracker.core.exceptions import PoseEstimationError
from vert_tracker.core.logging import get_logger
from vert_tracker.core.types import Frame, Landmark, Pose

if TYPE_CHECKING:
    from mediapipe.tasks.python.vision import PoseLandmarker
    from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult

logger = get_logger(__name__)

# Model download URL and local path
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_FILENAME = "pose_landmarker_heavy.task"


def resolve_project_root() -> Path:
    """Locate repository root by walking up from this file to pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: vision/ -> vert_tracker/ -> src/ -> repo
    return here.parents[3]


def default_model_dir() -> Path:
    """Return the project-local model cache directory under data/models."""
    return resolve_project_root() / "data" / "models"


def landmark_visibility(lm: object) -> float:
    """Extract a landmark's visibility score, preserving a true zero.

    MediaPipe landmarks report visibility as a float where 0.0 is a
    meaningful "not visible" signal. A naive `getattr(...) or 1.0` treats
    0.0 as falsy and silently promotes it to fully visible, which corrupts
    confidence calculations. This helper only defaults to 1.0 when the
    attribute is genuinely absent (e.g. bare landmarks with no visibility
    field), and otherwise clamps to the valid [0, 1] range.

    Args:
        lm: Landmark-like object, expected to optionally expose `visibility`

    Returns:
        Visibility score clamped to [0.0, 1.0]
    """
    raw = getattr(lm, "visibility", None)
    if raw is None:
        return 1.0
    value = float(raw)
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _download_model() -> Path:
    """Download the pose landmarker model if not present.

    Returns:
        Path to the downloaded model file

    Raises:
        PoseEstimationError: If download fails
    """
    model_dir = default_model_dir()
    model_path = model_dir / MODEL_FILENAME
    if model_path.exists():
        return model_path

    logger.info("Downloading MediaPipe pose landmarker model...")
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
        logger.info("Model downloaded to %s", model_path)
        return model_path
    except Exception as e:
        raise PoseEstimationError(f"Failed to download model: {e}") from e


class PoseEstimator:
    """Wrapper for MediaPipe pose estimation using the Tasks API.

    Converts MediaPipe results to custom Pose/Landmark types
    to avoid leaking MediaPipe objects throughout the codebase.
    """

    def __init__(self, settings: PoseSettings | None = None) -> None:
        """Initialize pose estimator with settings.

        Args:
            settings: Pose estimation settings (uses defaults if None)
        """
        self.settings = settings or PoseSettings()
        self._landmarker: PoseLandmarker | None = None  # pyright: ignore[reportInvalidTypeForm]
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if MediaPipe model is loaded."""
        return self._initialized

    def initialize(self) -> None:
        """Load MediaPipe pose model.

        Raises:
            PoseEstimationError: If model fails to load
        """
        try:
            model_path = _download_model()

            base_options = python.BaseOptions(model_asset_path=str(model_path))

            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self.settings.min_detection_confidence,
                min_pose_presence_confidence=self.settings.min_tracking_confidence,
                min_tracking_confidence=self.settings.min_tracking_confidence,
                output_segmentation_masks=self.settings.enable_segmentation,
            )

            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self._initialized = True
            logger.info("MediaPipe PoseLandmarker initialized (Tasks API)")

        except PoseEstimationError:
            raise
        except Exception as e:
            raise PoseEstimationError(f"Failed to initialize MediaPipe: {e}") from e

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            self._initialized = False

    def estimate(self, frame: Frame) -> Pose | None:
        """Run pose estimation on a frame.

        Args:
            frame: Input video frame

        Returns:
            Pose object with landmarks, or None if no pose detected

        Raises:
            PoseEstimationError: If estimation fails
        """
        if not self._initialized or self._landmarker is None:
            self.initialize()

        if self._landmarker is None:
            raise PoseEstimationError("Pose estimator not initialized")

        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)  # pyright: ignore[reportArgumentType]

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # Get timestamp in milliseconds
            timestamp_ms = int(frame.timestamp * 1000)

            # Run pose detection
            results = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            if not results.pose_landmarks or len(results.pose_landmarks) == 0:
                return None

            return self._convert_results(results, frame)

        except Exception as e:
            logger.error("Pose estimation failed: %s", e)
            raise PoseEstimationError(f"Estimation failed: {e}") from e

    def _convert_results(self, results: PoseLandmarkerResult, frame: Frame) -> Pose:
        """Convert MediaPipe results to Pose dataclass.

        Args:
            results: MediaPipe pose landmarker results
            frame: Original frame for metadata

        Returns:
            Pose object with converted landmarks
        """
        landmarks: dict[int, Landmark] = {}

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return Pose(
                landmarks={},
                timestamp=frame.timestamp,
                frame_idx=frame.index,
                confidence=0.0,
            )

        # Use first detected pose
        pose_landmarks = results.pose_landmarks[0]

        total_visibility = 0.0
        for idx, lm in enumerate(pose_landmarks):
            visibility = landmark_visibility(lm)
            landmarks[idx] = Landmark(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=visibility,
            )
            total_visibility += visibility

        # Average visibility as confidence proxy
        confidence = total_visibility / len(pose_landmarks) if pose_landmarks else 0.0

        return Pose(
            landmarks=landmarks,
            timestamp=frame.timestamp,
            frame_idx=frame.index,
            confidence=confidence,
        )

    def estimate_batch(self, frames: list[Frame]) -> list[Pose | None]:
        """Run pose estimation on multiple frames.

        Args:
            frames: List of input frames

        Returns:
            List of Pose objects (or None for failed detections)
        """
        return [self.estimate(frame) for frame in frames]

    def get_landmark_names(self) -> dict[int, str]:
        """Get mapping of landmark indices to names.

        Returns:
            Dictionary mapping index to landmark name
        """
        # Standard MediaPipe pose landmark names (33 landmarks)
        landmark_names = {
            0: "NOSE",
            1: "LEFT_EYE_INNER",
            2: "LEFT_EYE",
            3: "LEFT_EYE_OUTER",
            4: "RIGHT_EYE_INNER",
            5: "RIGHT_EYE",
            6: "RIGHT_EYE_OUTER",
            7: "LEFT_EAR",
            8: "RIGHT_EAR",
            9: "MOUTH_LEFT",
            10: "MOUTH_RIGHT",
            11: "LEFT_SHOULDER",
            12: "RIGHT_SHOULDER",
            13: "LEFT_ELBOW",
            14: "RIGHT_ELBOW",
            15: "LEFT_WRIST",
            16: "RIGHT_WRIST",
            17: "LEFT_PINKY",
            18: "RIGHT_PINKY",
            19: "LEFT_INDEX",
            20: "RIGHT_INDEX",
            21: "LEFT_THUMB",
            22: "RIGHT_THUMB",
            23: "LEFT_HIP",
            24: "RIGHT_HIP",
            25: "LEFT_KNEE",
            26: "RIGHT_KNEE",
            27: "LEFT_ANKLE",
            28: "RIGHT_ANKLE",
            29: "LEFT_HEEL",
            30: "RIGHT_HEEL",
            31: "LEFT_FOOT_INDEX",
            32: "RIGHT_FOOT_INDEX",
        }
        return landmark_names

    def __enter__(self) -> PoseEstimator:
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.close()
