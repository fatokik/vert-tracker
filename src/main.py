"""Main entry point for Vert Tracker application."""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np
from numpy.typing import NDArray

from core.config import get_settings
from core.exceptions import DroneConnectionError, VertTrackerError
from core.logging import get_logger, setup_logging
from drone.controller import TelloController
from drone.stream import VideoStream
from pipeline.processor import FrameProcessor
from ui.display import DisplayWindow, KeyAction
from ui.hud import HUDRenderer

logger = get_logger(__name__)


def run_tracking_session() -> int:
    """Run the main tracking session.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    settings = get_settings()
    setup_logging(settings.logging.level, settings.logging.file)

    logger.info("Starting Vert Tracker")

    # Initialize components
    controller = TelloController(settings.drone)
    display = DisplayWindow(settings.ui)
    processor = FrameProcessor(settings)
    hud = HUDRenderer(settings.ui)

    try:
        # Connect to drone
        logger.info("Connecting to Tello drone...")
        controller.connect()
        battery = controller.get_battery()
        logger.info("Connected (battery: %d%%)", battery)

        if battery < 10:
            logger.warning("Low battery! Consider charging before flight.")

        # Start video stream
        controller.start_stream()
        stream = VideoStream(controller)

        # Initialize processor
        processor.initialize()

        # Open display
        display.open()
        display.show_message("Connected! Press SPACE to start...", duration_ms=0)

        # Main processing loop
        frame_count = 0
        start_time = time.time()
        fps = 0.0

        logger.info("Starting tracking loop (press 'q' to quit)")

        with stream:
            for frame in stream.frames():
                # Process frame
                result = processor.process_frame(frame)

                # Render HUD
                output = hud.render_full_hud(
                    result.rendered_image,
                    stats=processor.stats,
                    phase=result.phase.name,
                    battery=controller.get_battery(),
                    fps=fps,
                )

                # Display frame
                display.show_frame(output)

                # Handle input
                action = display.poll_key(wait_ms=1)

                if action == KeyAction.QUIT:
                    logger.info("Quit requested")
                    break

                elif action == KeyAction.CALIBRATE:
                    logger.info("Calibration requested")
                    try:
                        processor.calibrate_with_aruco(frame)
                        display.show_message("Calibration successful!", duration_ms=1500)
                    except Exception as e:
                        logger.error("Calibration failed: %s", e)
                        display.show_message(f"Calibration failed: {e}", duration_ms=2000)

                elif action == KeyAction.RESET:
                    logger.info("Session reset requested")
                    processor.reset_session()
                    display.show_message("Session reset!", duration_ms=1000)

                elif action == KeyAction.SAVE:
                    logger.info("Save requested")
                    # TODO: Implement session saving
                    display.show_message("Session saved!", duration_ms=1000)

                elif action == KeyAction.PAUSE:
                    if display.is_paused:
                        display.show_message("PAUSED - Press SPACE to resume", duration_ms=0)

                # Update FPS
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()

        # Session complete
        summary = f"Session: {processor.stats.jump_count} jumps"
        if processor.stats.max_height:
            summary += f", max {processor.stats.max_height:.1f} cm"
        logger.info(summary)

        return 0

    except DroneConnectionError as e:
        logger.error("Drone connection failed: %s", e)
        return 1

    except VertTrackerError as e:
        logger.error("Tracking error: %s", e)
        return 2

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0

    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 3

    finally:
        # Cleanup
        processor.shutdown()
        display.close()
        controller.disconnect()
        logger.info("Vert Tracker stopped")


def run_demo_mode() -> int:
    """Run in demo mode without drone (for testing UI).

    Uses webcam or generates synthetic frames.

    Returns:
        Exit code
    """
    from core.types import Frame

    settings = get_settings()
    setup_logging(settings.logging.level)

    logger.info("Starting demo mode (no drone)")

    display = DisplayWindow(settings.ui)
    processor = FrameProcessor(settings)
    hud = HUDRenderer(settings.ui)

    # Try webcam
    cap_temp = cv2.VideoCapture(0)
    cap: cv2.VideoCapture | None
    if not cap_temp.isOpened():
        logger.warning("No webcam found, using synthetic frames")
        cap_temp.release()
        cap = None
    else:
        cap = cap_temp

    try:
        processor.initialize()
        display.open()

        frame_idx = 0
        start_time = time.time()
        fps = 0.0
        frame_count = 0

        while True:
            # Get frame
            image: NDArray[np.uint8]
            if cap is not None:
                ret, raw_image = cap.read()
                if not ret:
                    break
                image = np.asarray(raw_image, dtype=np.uint8)
            else:
                # Synthetic frame
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(
                    image,
                    "Demo Mode - No Camera",
                    (400, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    2,
                )

            frame = Frame(
                image=image,
                timestamp=time.time() - start_time,
                index=frame_idx,
            )
            frame_idx += 1

            # Process
            result = processor.process_frame(frame)

            # Render HUD
            output = hud.render_full_hud(
                result.rendered_image,
                stats=processor.stats,
                phase=result.phase.name,
                fps=fps,
            )

            display.show_frame(output)

            # Handle input
            action = display.poll_key(wait_ms=1)
            if action == KeyAction.QUIT:
                break

            # Update FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if frame_count >= 30:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

        return 0

    except Exception as e:
        logger.exception("Demo mode error: %s", e)
        return 1

    finally:
        if cap is not None:
            cap.release()
        processor.shutdown()
        display.close()


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Vert Tracker - Vertical jump measurement with drone and CV"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode without drone (uses webcam)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        import os

        os.environ["LOG_LEVEL"] = "DEBUG"

    exit_code = run_demo_mode() if args.demo else run_tracking_session()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
