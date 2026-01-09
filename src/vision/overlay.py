"""Overlay rendering for skeleton, metrics, and visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from core.config import UISettings
from core.types import Frame, JumpEvent, LandmarkIndex, Pose

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Skeleton connection pairs (landmark indices)
POSE_CONNECTIONS = [
    # Torso
    (LandmarkIndex.LEFT_SHOULDER.value, LandmarkIndex.RIGHT_SHOULDER.value),
    (LandmarkIndex.LEFT_SHOULDER.value, LandmarkIndex.LEFT_HIP.value),
    (LandmarkIndex.RIGHT_SHOULDER.value, LandmarkIndex.RIGHT_HIP.value),
    (LandmarkIndex.LEFT_HIP.value, LandmarkIndex.RIGHT_HIP.value),
    # Left leg
    (LandmarkIndex.LEFT_HIP.value, LandmarkIndex.LEFT_KNEE.value),
    (LandmarkIndex.LEFT_KNEE.value, LandmarkIndex.LEFT_ANKLE.value),
    (LandmarkIndex.LEFT_ANKLE.value, LandmarkIndex.LEFT_HEEL.value),
    (LandmarkIndex.LEFT_HEEL.value, LandmarkIndex.LEFT_FOOT_INDEX.value),
    (LandmarkIndex.LEFT_ANKLE.value, LandmarkIndex.LEFT_FOOT_INDEX.value),
    # Right leg
    (LandmarkIndex.RIGHT_HIP.value, LandmarkIndex.RIGHT_KNEE.value),
    (LandmarkIndex.RIGHT_KNEE.value, LandmarkIndex.RIGHT_ANKLE.value),
    (LandmarkIndex.RIGHT_ANKLE.value, LandmarkIndex.RIGHT_HEEL.value),
    (LandmarkIndex.RIGHT_HEEL.value, LandmarkIndex.RIGHT_FOOT_INDEX.value),
    (LandmarkIndex.RIGHT_ANKLE.value, LandmarkIndex.RIGHT_FOOT_INDEX.value),
]

# Colors (BGR format)
COLOR_SKELETON = (0, 255, 0)  # Green
COLOR_LANDMARK = (255, 255, 255)  # White
COLOR_HIP_CENTER = (0, 255, 255)  # Yellow
COLOR_TRAJECTORY = (255, 0, 255)  # Magenta
COLOR_TEXT = (255, 255, 255)  # White
COLOR_TEXT_BG = (0, 0, 0)  # Black


class OverlayRenderer:
    """Renders visual overlays on video frames.

    Handles skeleton drawing, trajectory visualization, and metric display.
    """

    def __init__(self, settings: UISettings | None = None) -> None:
        """Initialize renderer with settings.

        Args:
            settings: UI/display settings (uses defaults if None)
        """
        self.settings = settings or UISettings()

    def draw_skeleton(
        self,
        frame: Frame,
        pose: Pose,
        color: tuple[int, int, int] = COLOR_SKELETON,
        thickness: int = 2,
    ) -> NDArray[np.uint8]:
        """Draw pose skeleton on frame.

        Args:
            frame: Input frame
            pose: Detected pose with landmarks
            color: Line color (BGR)
            thickness: Line thickness

        Returns:
            Frame with skeleton overlay
        """
        image = frame.image.copy()
        width, height = frame.dimensions

        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            start_lm = pose.landmarks.get(start_idx)
            end_lm = pose.landmarks.get(end_idx)

            if start_lm is None or end_lm is None:
                continue

            if start_lm.visibility < 0.5 or end_lm.visibility < 0.5:
                continue

            start_pt = start_lm.to_pixel(width, height)
            end_pt = end_lm.to_pixel(width, height)

            cv2.line(image, start_pt, end_pt, color, thickness)

        # Draw landmarks
        for landmark in pose.landmarks.values():
            if landmark.visibility < 0.5:
                continue

            pt = landmark.to_pixel(width, height)
            cv2.circle(image, pt, 4, COLOR_LANDMARK, -1)

        # Highlight hip center
        hip_center = pose.hip_center
        if hip_center is not None and hip_center.visibility > 0.5:
            pt = hip_center.to_pixel(width, height)
            cv2.circle(image, pt, 8, COLOR_HIP_CENTER, -1)

        return image

    def draw_trajectory(
        self,
        image: NDArray[np.uint8],
        trajectory: list[tuple[int, float]],
        frame_height: int,
        color: tuple[int, int, int] = COLOR_TRAJECTORY,
        thickness: int = 2,
    ) -> NDArray[np.uint8]:
        """Draw jump trajectory on frame.

        Args:
            image: Input image array
            trajectory: List of (frame_idx, hip_y_normalized) points
            frame_height: Frame height for denormalization
            color: Line color (BGR)
            thickness: Line thickness

        Returns:
            Image with trajectory overlay
        """
        if len(trajectory) < 2:
            return image

        result = image.copy()

        # Convert to pixel coordinates
        # X position based on frame index, Y from normalized hip position
        x_scale = 5  # Pixels per frame
        x_offset = 50

        points = []
        for frame_idx, hip_y in trajectory:
            x = x_offset + (frame_idx - trajectory[0][0]) * x_scale
            y = int(hip_y * frame_height)
            points.append((x, y))

        # Draw trajectory line
        for i in range(len(points) - 1):
            cv2.line(result, points[i], points[i + 1], color, thickness)

        # Mark peak (minimum y = highest point)
        if points:
            peak_idx = min(range(len(points)), key=lambda i: points[i][1])
            cv2.circle(result, points[peak_idx], 6, (0, 0, 255), -1)

        return result

    def draw_metrics(
        self,
        image: NDArray[np.uint8],
        current_jump: JumpEvent | None = None,
        max_height: float | None = None,
        avg_height: float | None = None,
        jump_count: int = 0,
    ) -> NDArray[np.uint8]:
        """Draw metrics panel on frame.

        Args:
            image: Input image array
            current_jump: Current or last jump event
            max_height: Session max jump height
            avg_height: Session average jump height
            jump_count: Total jumps in session

        Returns:
            Image with metrics overlay
        """
        result = image.copy()

        # Metrics panel position
        x, y = 10, 30
        line_height = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7

        metrics = [
            f"Jumps: {jump_count}",
        ]

        if current_jump is not None:
            metrics.append(f"Last: {current_jump.height_cm:.1f} cm")

        if max_height is not None:
            metrics.append(f"Max: {max_height:.1f} cm")

        if avg_height is not None:
            metrics.append(f"Avg: {avg_height:.1f} cm")

        for i, text in enumerate(metrics):
            pos = (x, y + i * line_height)
            # Background for readability
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 2)
            cv2.rectangle(
                result,
                (pos[0] - 2, pos[1] - text_h - 2),
                (pos[0] + text_w + 2, pos[1] + 4),
                COLOR_TEXT_BG,
                -1,
            )
            cv2.putText(result, text, pos, font, font_scale, COLOR_TEXT, 2)

        return result

    def draw_phase_indicator(
        self,
        image: NDArray[np.uint8],
        phase: str,
    ) -> NDArray[np.uint8]:
        """Draw current jump phase indicator.

        Args:
            image: Input image array
            phase: Current phase name

        Returns:
            Image with phase indicator
        """
        result = image.copy()
        height, width = result.shape[:2]

        # Phase indicator in top-right corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0

        # Color based on phase
        phase_colors = {
            "IDLE": (128, 128, 128),  # Gray
            "TAKEOFF": (0, 255, 255),  # Yellow
            "AIRBORNE": (0, 255, 0),  # Green
            "LANDING": (0, 165, 255),  # Orange
        }
        color = phase_colors.get(phase, COLOR_TEXT)

        text = f"Phase: {phase}"
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 2)

        x = width - text_w - 20
        y = 40

        cv2.rectangle(
            result,
            (x - 5, y - text_h - 5),
            (x + text_w + 5, y + 5),
            COLOR_TEXT_BG,
            -1,
        )
        cv2.putText(result, text, (x, y), font, font_scale, color, 2)

        return result

    def draw_calibration_overlay(
        self,
        image: NDArray[np.uint8],
        px_per_cm: float,
        method: str,
    ) -> NDArray[np.uint8]:
        """Draw calibration status indicator.

        Args:
            image: Input image array
            px_per_cm: Current calibration value
            method: Calibration method name

        Returns:
            Image with calibration overlay
        """
        result = image.copy()
        height = result.shape[0]

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Cal: {px_per_cm:.2f} px/cm ({method})"

        pos = (10, height - 20)
        cv2.putText(result, text, pos, font, 0.5, COLOR_TEXT, 1)

        return result

    def render_full_overlay(
        self,
        frame: Frame,
        pose: Pose | None,
        phase: str = "IDLE",
        current_jump: JumpEvent | None = None,
        max_height: float | None = None,
        avg_height: float | None = None,
        jump_count: int = 0,
        trajectory: list[tuple[int, float]] | None = None,
    ) -> NDArray[np.uint8]:
        """Render complete overlay with all enabled elements.

        Args:
            frame: Input frame
            pose: Current pose (optional)
            phase: Current jump phase
            current_jump: Current/last jump event
            max_height: Session max height
            avg_height: Session average height
            jump_count: Total jump count
            trajectory: Jump trajectory points

        Returns:
            Fully rendered frame
        """
        image = frame.image.copy()

        # Skeleton overlay
        if self.settings.show_skeleton and pose is not None:
            image = self.draw_skeleton(
                Frame(image=image, timestamp=frame.timestamp, index=frame.index),
                pose,
            )

        # Trajectory overlay
        if self.settings.show_trajectory and trajectory:
            image = self.draw_trajectory(image, trajectory, frame.height)

        # Metrics panel
        if self.settings.show_metrics:
            image = self.draw_metrics(
                image,
                current_jump=current_jump,
                max_height=max_height,
                avg_height=avg_height,
                jump_count=jump_count,
            )

        # Phase indicator
        if self.settings.show_debug_info:
            image = self.draw_phase_indicator(image, phase)

        return image
