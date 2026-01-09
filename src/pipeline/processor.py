"""Frame processing pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from analysis.calculator import HeightCalculator
from analysis.detector import JumpDetector
from analysis.metrics import MetricsTracker
from core.config import Settings, get_settings
from core.logging import get_logger
from core.types import (
    CalibrationProfile,
    Frame,
    JumpEvent,
    JumpPhase,
    Pose,
    SessionStats,
)
from vision.calibration import Calibrator
from vision.filters import LandmarkSmoother
from vision.overlay import OverlayRenderer
from vision.pose import PoseEstimator

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


@dataclass
class ProcessedFrame:
    """Result of processing a single frame."""

    frame: Frame
    pose: Pose | None
    phase: JumpPhase
    jump_event: JumpEvent | None
    rendered_image: NDArray[np.uint8]


class FrameProcessor:
    """Orchestrates the full frame processing pipeline.

    Coordinates:
    - Pose estimation
    - Landmark smoothing
    - Jump detection
    - Height calculation
    - Overlay rendering
    """

    def __init__(
        self,
        settings: Settings | None = None,
        calibration: CalibrationProfile | None = None,
    ) -> None:
        """Initialize processor with settings.

        Args:
            settings: Application settings (uses defaults if None)
            calibration: Initial calibration profile
        """
        self.settings = settings or get_settings()

        # Components
        self._pose_estimator = PoseEstimator(self.settings.pose)
        self._smoother = LandmarkSmoother(self.settings.filter)
        self._detector = JumpDetector(self.settings.jump)
        self._calibrator = Calibrator(self.settings.calibration)
        self._overlay = OverlayRenderer(self.settings.ui)
        self._metrics = MetricsTracker()

        # State
        self._calibration = calibration or self._calibrator.get_default_profile()
        self._calculator = HeightCalculator(self._calibration)
        self._current_trajectory: list[tuple[int, float]] = []
        self._initialized = False

    @property
    def stats(self) -> SessionStats:
        """Get current session statistics."""
        return self._metrics.stats

    @property
    def calibration(self) -> CalibrationProfile:
        """Get current calibration profile."""
        return self._calibration

    @property
    def current_phase(self) -> JumpPhase:
        """Get current jump phase."""
        return self._detector.current_phase

    def initialize(self) -> None:
        """Initialize all components."""
        if not self._initialized:
            self._pose_estimator.initialize()
            self._initialized = True
            logger.info("Frame processor initialized")

    def shutdown(self) -> None:
        """Release all resources."""
        self._pose_estimator.close()
        self._initialized = False
        logger.info("Frame processor shutdown")

    def set_calibration(self, calibration: CalibrationProfile) -> None:
        """Update calibration profile.

        Args:
            calibration: New calibration profile
        """
        self._calibration = calibration
        self._calculator = HeightCalculator(calibration)
        self._metrics.stats.calibration = calibration
        logger.info("Calibration updated: %.2f px/cm", calibration.px_per_cm)

    def process_frame(self, frame: Frame) -> ProcessedFrame:
        """Process a single frame through the full pipeline.

        Args:
            frame: Input video frame

        Returns:
            ProcessedFrame with all results
        """
        if not self._initialized:
            self.initialize()

        # Step 1: Pose estimation
        pose = self._pose_estimator.estimate(frame)

        # Step 2: Smooth landmarks (if pose detected)
        if pose is not None:
            pose = self._smooth_pose(pose)

        # Step 3: Jump detection
        jump_event = None
        if pose is not None:
            jump_event = self._detector.update(pose)

            # Track trajectory for visualization
            hip_center = pose.hip_center
            if hip_center is not None:
                if self._detector.is_jumping:
                    self._current_trajectory.append((frame.index, hip_center.y))
                else:
                    self._current_trajectory.clear()

        # Step 4: Calculate height (if jump completed)
        if jump_event is not None:
            height = self._calculator.calculate_height(jump_event, frame.height)
            # Create new event with calculated height
            jump_event = JumpEvent(
                takeoff_frame=jump_event.takeoff_frame,
                peak_frame=jump_event.peak_frame,
                landing_frame=jump_event.landing_frame,
                height_cm=height,
                confidence=jump_event.confidence,
                peak_hip_y=jump_event.peak_hip_y,
                baseline_hip_y=jump_event.baseline_hip_y,
                trajectory=jump_event.trajectory,
            )
            self._metrics.add_jump(jump_event)
            logger.info(
                "Jump detected: %.1f cm (frames %d-%d)",
                height,
                jump_event.takeoff_frame,
                jump_event.landing_frame,
            )

        # Step 5: Render overlay
        rendered = self._overlay.render_full_overlay(
            frame=frame,
            pose=pose,
            phase=self._detector.current_phase.name,
            current_jump=self._metrics.last_jump,
            max_height=self._metrics.max_height,
            avg_height=self._metrics.avg_height,
            jump_count=self._metrics.jump_count,
            trajectory=self._current_trajectory if self._detector.is_jumping else None,
        )

        return ProcessedFrame(
            frame=frame,
            pose=pose,
            phase=self._detector.current_phase,
            jump_event=jump_event,
            rendered_image=rendered,
        )

    def _smooth_pose(self, pose: Pose) -> Pose:
        """Apply Kalman filtering to pose landmarks.

        Args:
            pose: Raw pose data

        Returns:
            Pose with smoothed landmark positions
        """
        from core.types import Landmark

        smoothed_landmarks: dict[int, Landmark] = {}

        for idx, landmark in pose.landmarks.items():
            smooth_x, smooth_y = self._smoother.smooth(idx, landmark.x, landmark.y)
            smoothed_landmarks[idx] = Landmark(
                x=smooth_x,
                y=smooth_y,
                z=landmark.z,
                visibility=landmark.visibility,
            )

        return Pose(
            landmarks=smoothed_landmarks,
            timestamp=pose.timestamp,
            frame_idx=pose.frame_idx,
            confidence=pose.confidence,
        )

    def calibrate_with_aruco(self, frame: Frame) -> CalibrationProfile:
        """Run ArUco calibration on frame.

        Args:
            frame: Frame containing ArUco marker

        Returns:
            New calibration profile
        """
        profile = self._calibrator.calibrate_with_aruco(frame)
        self.set_calibration(profile)
        return profile

    def calibrate_with_height(
        self,
        frame: Frame,
        pose: Pose,
        known_height_cm: float,
    ) -> CalibrationProfile:
        """Run height-based calibration.

        Args:
            frame: Current frame
            pose: Pose with visible head and feet
            known_height_cm: Person's actual height

        Returns:
            New calibration profile
        """
        # Get head and feet positions
        from core.types import LandmarkIndex

        nose = pose.get_landmark(LandmarkIndex.NOSE)
        left_ankle = pose.get_landmark(LandmarkIndex.LEFT_ANKLE)
        right_ankle = pose.get_landmark(LandmarkIndex.RIGHT_ANKLE)

        if nose is None or (left_ankle is None and right_ankle is None):
            from core.exceptions import CalibrationError

            raise CalibrationError("Cannot detect head and feet for calibration")

        head_y = nose.y
        feet_y = max(
            left_ankle.y if left_ankle else 0,
            right_ankle.y if right_ankle else 0,
        )

        profile = self._calibrator.calibrate_with_height(frame, head_y, feet_y, known_height_cm)
        self.set_calibration(profile)
        return profile

    def reset_session(self) -> None:
        """Reset session statistics."""
        self._metrics.reset()
        self._detector.reset()
        self._smoother.reset()
        self._current_trajectory.clear()
        logger.info("Session reset")

    def __enter__(self) -> FrameProcessor:
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
        self.shutdown()
