#!/usr/bin/env python3
"""Record raw video for offline testing and validation.

Records video from the Tello drone (or webcam in demo mode)
for later analysis and testing without the drone.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from vert_tracker.core.config import get_settings
from vert_tracker.core.logging import get_logger, setup_logging
from vert_tracker.drone.controller import TelloController
from vert_tracker.drone.stream import VideoStream

logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/recordings")


def record_from_drone(
    controller: TelloController,
    output_path: Path,
    duration_seconds: float | None = None,
    fps: float = 30.0,
) -> int:
    """Record video from drone.

    Args:
        controller: Connected drone controller
        output_path: Output video file path
        duration_seconds: Maximum recording duration (None = until 'q' pressed)
        fps: Output video frame rate

    Returns:
        Number of frames recorded
    """
    stream = VideoStream(controller)

    # Get first frame to determine dimensions
    stream.start()
    first_frame = stream.capture_single()

    if first_frame is None:
        logger.error("Could not capture initial frame")
        return 0

    width, height = first_frame.dimensions

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        logger.error("Could not open video writer")
        return 0

    logger.info(
        "Recording to %s (%dx%d @ %.1f fps)",
        output_path,
        width,
        height,
        fps,
    )
    logger.info("Press 'q' to stop recording")

    cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()

    try:
        for frame in stream.frames():
            # Write frame
            writer.write(frame.image)
            frame_count += 1

            # Display preview
            display = frame.image.copy()
            elapsed = time.time() - start_time

            status = f"Recording: {frame_count} frames | {elapsed:.1f}s"
            cv2.putText(
                display,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            cv2.circle(display, (width - 30, 30), 15, (0, 0, 255), -1)  # Record indicator

            cv2.imshow("Recording", display)

            # Check for stop conditions
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Recording stopped by user")
                break

            if duration_seconds and elapsed >= duration_seconds:
                logger.info("Recording duration reached")
                break

    finally:
        writer.release()
        stream.stop()
        cv2.destroyAllWindows()

    logger.info("Recorded %d frames to %s", frame_count, output_path)
    return frame_count


def record_from_webcam(
    output_path: Path,
    duration_seconds: float | None = None,
    fps: float = 30.0,
    camera_id: int = 0,
) -> int:
    """Record video from webcam.

    Args:
        output_path: Output video file path
        duration_seconds: Maximum recording duration
        fps: Output video frame rate
        camera_id: Webcam device ID

    Returns:
        Number of frames recorded
    """
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        logger.error("Could not open webcam %d", camera_id)
        return 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    logger.info(
        "Recording from webcam to %s (%dx%d @ %.1f fps)",
        output_path,
        width,
        height,
        fps,
    )
    logger.info("Press 'q' to stop recording")

    cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            writer.write(frame)
            frame_count += 1

            elapsed = time.time() - start_time

            # Display preview
            display = frame.copy()
            status = f"Recording: {frame_count} frames | {elapsed:.1f}s"
            cv2.putText(
                display,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            cv2.circle(display, (width - 30, 30), 15, (0, 0, 255), -1)

            cv2.imshow("Recording", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if duration_seconds and elapsed >= duration_seconds:
                break

    finally:
        writer.release()
        cap.release()
        cv2.destroyAllWindows()

    logger.info("Recorded %d frames to %s", frame_count, output_path)
    return frame_count


def main() -> int:
    """Run recording script."""
    parser = argparse.ArgumentParser(description="Record video for offline testing")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output video file path (default: auto-generated)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        help="Maximum recording duration in seconds",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output frame rate (default: 30)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use webcam instead of drone",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device ID (default: 0)",
    )

    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.logging.level)

    # Generate output path if not specified
    if args.output is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = DEFAULT_OUTPUT_DIR / f"session_{timestamp}.mp4"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.demo:
        frame_count = record_from_webcam(
            args.output,
            duration_seconds=args.duration,
            fps=args.fps,
            camera_id=args.camera,
        )
    else:
        controller = TelloController(settings.drone)
        try:
            controller.connect()
            controller.start_stream()
            frame_count = record_from_drone(
                controller,
                args.output,
                duration_seconds=args.duration,
                fps=args.fps,
            )
        finally:
            controller.disconnect()

    return 0 if frame_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
