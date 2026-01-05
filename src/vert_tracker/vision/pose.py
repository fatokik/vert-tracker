"""MediaPipe pose estimation wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mediapipe as mp

from vert_tracker.core.config import PoseSettings
from vert_tracker.core.exceptions import PoseEstimationError
from vert_tracker.core.logging import get_logger
from vert_tracker.core.types import Frame, Landmark, Pose

if TYPE_CHECKING:
    from mediapipe.python.solutions.pose import Pose as MPPose

logger = get_logger(__name__)


class PoseEstimator:
    """Wrapper for MediaPipe pose estimation.

    Converts MediaPipe results to custom Pose/Landmark types
    to avoid leaking MediaPipe objects throughout the codebase.
    """

    def __init__(self, settings: PoseSettings | None = None) -> None:
        """Initialize pose estimator with settings.

        Args:
            settings: Pose estimation settings (uses defaults if None)
        """
        self.settings = settings or PoseSettings()
        self._pose: MPPose | None = None
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
            mp_pose = mp.solutions.pose
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.settings.model_complexity,
                enable_segmentation=self.settings.enable_segmentation,
                min_detection_confidence=self.settings.min_detection_confidence,
                min_tracking_confidence=self.settings.min_tracking_confidence,
            )
            self._initialized = True
            logger.info(
                "MediaPipe Pose initialized (complexity=%d)",
                self.settings.model_complexity,
            )

        except Exception as e:
            raise PoseEstimationError(f"Failed to initialize MediaPipe: {e}") from e

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None
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
        if not self._initialized or self._pose is None:
            self.initialize()

        # After initialize(), _pose should not be None
        if self._pose is None:
            raise PoseEstimationError("Pose estimator not initialized")

        try:
            # MediaPipe expects RGB
            import cv2

            rgb_image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb_image)

            if results.pose_landmarks is None:
                return None

            return self._convert_results(results, frame)

        except Exception as e:
            logger.error("Pose estimation failed: %s", e)
            raise PoseEstimationError(f"Estimation failed: {e}") from e

    def _convert_results(self, results: object, frame: Frame) -> Pose:
        """Convert MediaPipe results to Pose dataclass.

        Args:
            results: MediaPipe pose results
            frame: Original frame for metadata

        Returns:
            Pose object with converted landmarks
        """
        landmarks: dict[int, Landmark] = {}
        pose_landmarks = getattr(results, "pose_landmarks", None)

        if pose_landmarks is None:
            return Pose(
                landmarks={},
                timestamp=frame.timestamp,
                frame_idx=frame.index,
                confidence=0.0,
            )

        total_visibility = 0.0
        for idx, lm in enumerate(pose_landmarks.landmark):
            landmarks[idx] = Landmark(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility),
            )
            total_visibility += lm.visibility

        # Average visibility as confidence proxy
        if pose_landmarks.landmark:
            confidence = total_visibility / len(pose_landmarks.landmark)
        else:
            confidence = 0.0

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
        mp_pose = mp.solutions.pose
        return {lm.value: lm.name for lm in mp_pose.PoseLandmark}

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
