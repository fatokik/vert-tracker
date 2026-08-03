#!/usr/bin/env python3
"""Standalone calibration routine for Vert Tracker.

Run this script to calibrate the system before a training session.
Supports multiple calibration methods:
- ArUco marker detection
- Known height reference
- Manual pixel/cm specification
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from vert_tracker.core.config import get_settings
from vert_tracker.core.logging import get_logger, setup_logging
from vert_tracker.drone.controller import TelloController
from vert_tracker.drone.stream import VideoStream
from vert_tracker.vision.calibration import Calibrator
from vert_tracker.vision.pose import PoseEstimator

logger = get_logger(__name__)

DEFAULT_CALIBRATION_PATH = Path("data/calibration/profile.json")


def calibrate_with_aruco(
    calibrator: Calibrator,
    controller: TelloController,
) -> bool:
    """Run ArUco marker calibration.

    Args:
        calibrator: Calibrator instance
        controller: Connected drone controller

    Returns:
        True if calibration succeeded
    """
    logger.info("Starting ArUco calibration...")
    logger.info("Hold an ArUco marker (DICT_4X4_50) visible to the drone")

    stream = VideoStream(controller)

    try:
        stream.start()

        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        logger.info("Press 'c' when marker is visible, 'q' to cancel")

        for frame in stream.frames():
            # Check for markers
            markers = calibrator.detect_aruco_markers(frame)

            # Draw markers on frame
            display = frame.image.copy()
            for marker_id, corners in markers:
                pts = corners.astype(int)
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                center = pts.mean(axis=0).astype(int)
                cv2.putText(
                    display,
                    f"ID: {marker_id}",
                    tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Status text
            status = f"Markers detected: {len(markers)}"
            cv2.putText(
                display,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Calibration cancelled")
                return False
            elif key == ord("c") and markers:
                try:
                    profile = calibrator.calibrate_with_aruco(frame)
                    logger.info(
                        "Calibration successful: %.2f px/cm",
                        profile.px_per_cm,
                    )
                    return True
                except Exception as e:
                    logger.error("Calibration failed: %s", e)

    finally:
        stream.stop()
        cv2.destroyAllWindows()

    return False


def calibrate_with_height(
    calibrator: Calibrator,
    controller: TelloController,
    known_height_cm: float,
) -> bool:
    """Run height-based calibration.

    Args:
        calibrator: Calibrator instance
        controller: Connected drone controller
        known_height_cm: Person's known height in cm

    Returns:
        True if calibration succeeded
    """
    logger.info("Starting height calibration...")
    logger.info("Stand in frame with full body visible (head to toe)")

    stream = VideoStream(controller)
    pose_estimator = PoseEstimator()

    try:
        stream.start()
        pose_estimator.initialize()

        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        logger.info("Press 'c' when standing upright, 'q' to cancel")

        for frame in stream.frames():
            pose = pose_estimator.estimate(frame)

            display = frame.image.copy()

            if pose is not None:
                # Draw skeleton preview
                from vert_tracker.core.types import LandmarkIndex

                nose = pose.get_landmark(LandmarkIndex.NOSE)
                left_ankle = pose.get_landmark(LandmarkIndex.LEFT_ANKLE)
                right_ankle = pose.get_landmark(LandmarkIndex.RIGHT_ANKLE)

                if nose and (left_ankle or right_ankle):
                    head_pt = nose.to_pixel(frame.width, frame.height)
                    feet_y = max(
                        (left_ankle.y if left_ankle else 0),
                        (right_ankle.y if right_ankle else 0),
                    )
                    feet_pt = (head_pt[0], int(feet_y * frame.height))

                    cv2.circle(display, head_pt, 8, (0, 255, 0), -1)
                    cv2.circle(display, feet_pt, 8, (0, 255, 0), -1)
                    cv2.line(display, head_pt, feet_pt, (0, 255, 0), 2)

                    status = "Pose detected - Press 'c' to calibrate"
                else:
                    status = "Full body not visible"
            else:
                status = "No pose detected"

            cv2.putText(
                display,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Calibration cancelled")
                return False
            elif key == ord("c") and pose is not None:
                try:
                    from vert_tracker.core.types import LandmarkIndex

                    nose = pose.get_landmark(LandmarkIndex.NOSE)
                    left_ankle = pose.get_landmark(LandmarkIndex.LEFT_ANKLE)
                    right_ankle = pose.get_landmark(LandmarkIndex.RIGHT_ANKLE)

                    if nose and (left_ankle or right_ankle):
                        head_y = nose.y
                        feet_y = max(
                            (left_ankle.y if left_ankle else 0),
                            (right_ankle.y if right_ankle else 0),
                        )

                        profile = calibrator.calibrate_with_height(
                            frame, head_y, feet_y, known_height_cm
                        )
                        logger.info(
                            "Calibration successful: %.2f px/cm",
                            profile.px_per_cm,
                        )
                        return True
                except Exception as e:
                    logger.error("Calibration failed: %s", e)

    finally:
        stream.stop()
        pose_estimator.close()
        cv2.destroyAllWindows()

    return False


def main() -> int:
    """Run calibration routine."""
    parser = argparse.ArgumentParser(description="Calibrate Vert Tracker system")
    parser.add_argument(
        "--method",
        choices=["aruco", "height", "manual"],
        default="aruco",
        help="Calibration method (default: aruco)",
    )
    parser.add_argument(
        "--height",
        type=float,
        help="Known height in cm (required for height method)",
    )
    parser.add_argument(
        "--px-per-cm",
        type=float,
        help="Manual pixels per cm value",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="Output path for calibration profile",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use webcam instead of drone",
    )

    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.logging.level)

    calibrator = Calibrator(settings.calibration)

    # Manual calibration doesn't need drone
    if args.method == "manual":
        if args.px_per_cm is None:
            logger.error("--px-per-cm required for manual calibration")
            return 1

        calibrator.calibrate_manual(args.px_per_cm)
        calibrator.save_profile(args.output)
        logger.info("Saved calibration to %s", args.output)
        return 0

    # Height calibration requires height
    if args.method == "height" and args.height is None:
        logger.error("--height required for height calibration")
        return 1

    # Demo mode uses webcam
    if args.demo:
        logger.info("Demo mode: using webcam")
        # TODO: Implement webcam calibration
        logger.error("Webcam calibration not yet implemented")
        return 1

    # Connect to drone
    controller = TelloController(settings.drone)

    try:
        controller.connect()
        controller.start_stream()

        if args.method == "aruco":
            success = calibrate_with_aruco(calibrator, controller)
        else:  # height
            success = calibrate_with_height(calibrator, controller, args.height)

        if success:
            calibrator.save_profile(args.output)
            logger.info("Saved calibration to %s", args.output)
            return 0
        else:
            return 1

    except Exception as e:
        logger.exception("Calibration error: %s", e)
        return 1

    finally:
        controller.disconnect()


if __name__ == "__main__":
    sys.exit(main())
