"""Main entry point for Vert Tracker application."""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np
from numpy.typing import NDArray

from vert_tracker.core.config import get_settings
from vert_tracker.core.exceptions import DroneConnectionError, VertTrackerError
from vert_tracker.core.logging import get_logger, setup_logging
from vert_tracker.drone.controller import TelloController
from vert_tracker.drone.stream import VideoStream
from vert_tracker.pipeline.processor import FrameProcessor
from vert_tracker.ui.display import DisplayWindow, KeyAction
from vert_tracker.ui.hud import HUDRenderer

logger = get_logger(__name__)

# Flight control constants
MOVE_DISTANCE_CM = 30  # Distance per keypress
ROTATE_ANGLE_DEG = 15  # Rotation per keypress


def handle_flight_control(
    controller: TelloController,
    action: KeyAction,
    display: DisplayWindow,
    is_flying: bool,
) -> bool:
    """Execute flight control action on the drone.

    Args:
        controller: Drone controller
        action: The flight control action to execute
        display: Display window for error messages
        is_flying: Whether the drone is currently in flight

    Returns:
        Updated is_flying state
    """
    try:
        # Handle takeoff/land regardless of flight state
        if action == KeyAction.TAKEOFF:
            if is_flying:
                display.show_message("Already in flight!", duration_ms=1000)
            else:
                logger.info("Taking off...")
                display.show_message("Taking off...", duration_ms=500)
                controller.takeoff()
                logger.info("Takeoff complete")
                display.show_message("Airborne! Use movement keys.", duration_ms=1500)
                return True
            return is_flying

        if action == KeyAction.LAND:
            if not is_flying:
                display.show_message("Not in flight!", duration_ms=1000)
            else:
                logger.info("Landing...")
                display.show_message("Landing...", duration_ms=500)
                controller.land()
                logger.info("Landed")
                display.show_message("Landed safely.", duration_ms=1500)
                return False
            return is_flying

        # Movement commands require the drone to be flying
        if not is_flying:
            display.show_message("Press T to takeoff first!", duration_ms=1000)
            return is_flying

        if action == KeyAction.MOVE_FORWARD:
            controller.move_forward(MOVE_DISTANCE_CM)
        elif action == KeyAction.MOVE_BACKWARD:
            controller.move_backward(MOVE_DISTANCE_CM)
        elif action == KeyAction.MOVE_LEFT:
            controller.move_left(MOVE_DISTANCE_CM)
        elif action == KeyAction.MOVE_RIGHT:
            controller.move_right(MOVE_DISTANCE_CM)
        elif action == KeyAction.MOVE_UP:
            controller.move_up(MOVE_DISTANCE_CM)
        elif action == KeyAction.MOVE_DOWN:
            controller.move_down(MOVE_DISTANCE_CM)
        elif action == KeyAction.ROTATE_LEFT:
            controller.rotate_counter_clockwise(ROTATE_ANGLE_DEG)
        elif action == KeyAction.ROTATE_RIGHT:
            controller.rotate_clockwise(ROTATE_ANGLE_DEG)

    except Exception as e:
        logger.error("Flight control error: %s", e)
        display.show_message(f"Flight error: {e}", duration_ms=1000)

    return is_flying


def is_flight_action(action: KeyAction) -> bool:
    """Check if an action is a flight control action."""
    return action in {
        KeyAction.MOVE_FORWARD,
        KeyAction.MOVE_BACKWARD,
        KeyAction.MOVE_LEFT,
        KeyAction.MOVE_RIGHT,
        KeyAction.MOVE_UP,
        KeyAction.MOVE_DOWN,
        KeyAction.ROTATE_LEFT,
        KeyAction.ROTATE_RIGHT,
        KeyAction.TAKEOFF,
        KeyAction.LAND,
    }


def draw_mode_overlay(
    image: NDArray[np.uint8],
    is_positioning: bool,
    battery: int,
) -> NDArray[np.uint8]:
    """Draw mode indicator and controls help on the frame.

    Args:
        image: Frame to draw on
        is_positioning: Whether in positioning mode
        battery: Current battery percentage

    Returns:
        Frame with overlay drawn
    """
    output = image.copy()
    h, w = output.shape[:2]

    if is_positioning:
        # Positioning mode overlay
        mode_text = "POSITIONING MODE"
        mode_color = (0, 165, 255)  # Orange
        controls = [
            "T: Takeoff / L: Land",
            "W/X: Forward/Back",
            "A/E: Left/Right",
            "I/J: Up/Down",
            "U/O: Rotate L/R",
            "ENTER/P: Start Tracking",
            "Q: Quit",
        ]
    else:
        # Tracking mode overlay
        mode_text = "TRACKING MODE"
        mode_color = (0, 255, 0)  # Green
        controls = [
            "P: Reposition Drone",
            "C: Calibrate",
            "R: Reset Session",
            "Q: Quit",
        ]

    # Draw mode banner at top
    cv2.rectangle(output, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(
        output,
        mode_text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        mode_color,
        2,
    )

    # Draw battery
    battery_text = f"Battery: {battery}%"
    battery_color = (0, 255, 0) if battery > 20 else (0, 0, 255)
    cv2.putText(
        output,
        battery_text,
        (w - 150, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        battery_color,
        2,
    )

    # Draw controls help on the right side
    y_offset = 80
    for control in controls:
        cv2.putText(
            output,
            control,
            (w - 200, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        y_offset += 22

    return output


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
        display.show_message(
            "Connected! Press T to takeoff, then position with WASD keys",
            duration_ms=3000,
        )

        # Main processing loop
        frame_count = 0
        start_time = time.time()
        fps = 0.0
        is_positioning = True  # Start in positioning mode
        is_flying = False  # Track if drone is in flight

        logger.info("Starting in positioning mode - press T to takeoff, then use movement keys")

        with stream:
            for frame in stream.frames():
                battery = controller.get_battery()

                if is_positioning:
                    # Positioning mode: show live feed with flight controls overlay
                    output = draw_mode_overlay(frame.image, is_positioning=True, battery=battery)
                else:
                    # Tracking mode: full pose processing pipeline
                    result = processor.process_frame(frame)
                    output = hud.render_full_hud(
                        result.rendered_image,
                        stats=processor.stats,
                        phase=result.phase.name,
                        battery=battery,
                        fps=fps,
                    )
                    # Add tracking mode indicator
                    output = draw_mode_overlay(output, is_positioning=False, battery=battery)

                # Display frame
                display.show_frame(output)

                # Handle input
                action = display.poll_key(wait_ms=1)

                if action == KeyAction.QUIT:
                    logger.info("Quit requested")
                    break

                elif action == KeyAction.TOGGLE_POSITIONING:
                    is_positioning = not is_positioning
                    if is_positioning:
                        logger.info("Switched to positioning mode")
                        display.show_message("POSITIONING MODE", duration_ms=1000)
                    else:
                        logger.info("Switched to tracking mode")
                        display.show_message("TRACKING MODE - Position locked!", duration_ms=1000)

                elif is_positioning and is_flight_action(action):
                    # Handle flight controls only in positioning mode
                    is_flying = handle_flight_control(controller, action, display, is_flying)

                elif not is_positioning:
                    # Handle tracking mode actions
                    if action == KeyAction.CALIBRATE:
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
    from vert_tracker.core.types import Frame

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
        session_start_time = time.time()  # Never reset - used for frame timestamps
        fps_start_time = time.time()  # Reset periodically for FPS calculation
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
                timestamp=time.time() - session_start_time,
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
            elapsed = time.time() - fps_start_time
            if frame_count >= 30:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

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
