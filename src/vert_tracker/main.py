"""Main entry point for Vert Tracker application."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from vert_tracker.analysis.metrics import export_session_data
from vert_tracker.core.config import get_settings
from vert_tracker.core.exceptions import DroneConnectionError, VertTrackerError
from vert_tracker.core.logging import get_logger, setup_logging
from vert_tracker.core.types import SessionStats
from vert_tracker.drone.controller import TelloController
from vert_tracker.drone.flight_session import FlightSession
from vert_tracker.drone.stream import VideoStream
from vert_tracker.pipeline.processor import FrameProcessor
from vert_tracker.ui.display import DisplayWindow, KeyAction
from vert_tracker.ui.hud import HUDRenderer
from vert_tracker.vision.calibration import Calibrator, bootstrap_calibration

logger = get_logger(__name__)

DEFAULT_SESSION_DIR = Path("data/sessions")


def save_session_stats(stats: SessionStats, directory: Path | None = None) -> Path:
    """Persist session statistics to a timestamped JSON file.

    Args:
        stats: Session statistics to export
        directory: Output directory (defaults to `data/sessions`)

    Returns:
        Path to the written session file

    Raises:
        OSError: If the file could not be written
    """
    out_dir = directory or DEFAULT_SESSION_DIR
    path = out_dir / f"session_{time.strftime('%Y%m%d_%H%M%S')}.json"
    export_session_data(stats, path)
    return path


RC_STICK_ACTIONS = {
    KeyAction.MOVE_FORWARD,
    KeyAction.MOVE_BACKWARD,
    KeyAction.MOVE_LEFT,
    KeyAction.MOVE_RIGHT,
    KeyAction.MOVE_UP,
    KeyAction.MOVE_DOWN,
    KeyAction.ROTATE_LEFT,
    KeyAction.ROTATE_RIGHT,
}


def handle_flight_control(
    session: FlightSession,
    action: KeyAction,
    display: DisplayWindow,
) -> None:
    """Apply takeoff/land/RC stick input through FlightSession."""
    try:
        if action == KeyAction.TAKEOFF:
            if session.is_airborne:
                display.show_message("Already in flight!", duration_ms=1000)
            else:
                display.show_message("Taking off...", duration_ms=500)
                session.takeoff()
                display.show_message("Airborne! Hold movement keys (RC).", duration_ms=1500)
            return

        if action == KeyAction.LAND:
            if not session.is_airborne:
                display.show_message("Not in flight!", duration_ms=1000)
            else:
                display.show_message("Landing...", duration_ms=500)
                session.land()
                display.show_message("Landed safely.", duration_ms=1500)
            return

        if action in RC_STICK_ACTIONS:
            if not session.is_airborne:
                display.show_message("Press T to takeoff first!", duration_ms=1000)
                return
            session.apply_key_action(action)

    except Exception as e:
        logger.error("Flight control error: %s", e)
        display.show_message(f"Flight error: {e}", duration_ms=1000)


def is_flight_action(action: KeyAction) -> bool:
    """Check if an action is a flight control action."""
    return action in RC_STICK_ACTIONS | {KeyAction.TAKEOFF, KeyAction.LAND}


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
            "Hold W/X: Fwd/Back (RC)",
            "Hold A/E: Left/Right",
            "Hold I/J: Up/Down",
            "Hold U/O: Rotate",
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


def build_processor(
    calibration_path: Path | None = None,
) -> FrameProcessor:
    """Create a FrameProcessor with bootstrapped calibration."""
    settings = get_settings()
    calibrator = Calibrator(settings.calibration)
    profile, is_calibrated = bootstrap_calibration(calibrator, path=calibration_path)
    return FrameProcessor(
        settings,
        calibration=profile,
        is_calibrated=is_calibrated,
    )


def run_tracking_session(calibration_path: Path | None = None) -> int:
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
    processor = build_processor(calibration_path)
    hud = HUDRenderer(settings.ui)
    flight: FlightSession | None = None

    try:
        # Connect to drone
        logger.info("Connecting to Tello drone...")
        controller.connect()
        flight = FlightSession(controller)
        battery = flight.battery
        logger.info("Connected (battery: %d%%)", battery)

        if battery < 10:
            logger.warning("Low battery! Consider charging before flight.")

        # VideoStream.start() owns streamon — do not call start_stream here
        stream = VideoStream(controller)

        # Initialize processor
        processor.initialize()

        # Open display
        display.open()
        display.show_message(
            "Connected! Press T to takeoff, then hold W/X/A/E to position (RC)",
            duration_ms=3000,
        )

        # Main processing loop
        frame_count = 0
        start_time = time.time()
        fps = 0.0
        is_positioning = True  # Start in positioning mode

        logger.info("Starting in positioning mode - press T to takeoff, then hold RC keys")

        with stream:
            for frame in stream.frames():
                battery = flight.battery

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
                    flight.clear_sticks()
                    if is_positioning:
                        logger.info("Switched to positioning mode")
                        display.show_message("POSITIONING MODE", duration_ms=1000)
                    else:
                        logger.info("Switched to tracking mode")
                        display.show_message("TRACKING MODE - Position locked!", duration_ms=1000)

                elif is_positioning and is_flight_action(action):
                    handle_flight_control(flight, action, display)

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
                        try:
                            saved_path = save_session_stats(processor.stats)
                            display.show_message(
                                f"Saved {processor.stats.jump_count} jumps -> {saved_path}",
                                duration_ms=2000,
                            )
                        except OSError as e:
                            logger.exception("Session save failed")
                            display.show_message(f"Save failed: {e}", duration_ms=2000)

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
        processor.shutdown()
        display.close()
        if flight is not None:
            flight.shutdown_safe()
        else:
            controller.disconnect()
        logger.info("Vert Tracker stopped")


def _synthetic_demo_frame(width: int = 1280, height: int = 720) -> NDArray[np.uint8]:
    """Build a placeholder frame when no webcam frames are available."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        image,
        "Demo Mode - No Camera",
        (max(40, width // 2 - 280), height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        image,
        "Press q to quit",
        (max(40, width // 2 - 120), height // 2 + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        1,
    )
    return image


def _open_demo_camera(device_index: int = 0, warmup_reads: int = 30) -> cv2.VideoCapture | None:
    """Open a webcam and require at least one successful frame before use.

    macOS often reports VideoCapture as opened before frames are available
    (permissions race, Continuity Camera, etc.). A failed first read used to
    exit demo mode silently — warm up and fall back instead.
    """
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        logger.warning("No webcam found at index %d", device_index)
        cap.release()
        return None

    for attempt in range(warmup_reads):
        ret, _frame = cap.read()
        if ret:
            logger.info("Webcam ready after %d warmup read(s)", attempt + 1)
            return cap
        time.sleep(0.05)

    logger.warning(
        "Webcam opened but delivered no frames after %d attempts; using synthetic frames",
        warmup_reads,
    )
    cap.release()
    return None


def run_demo_mode(calibration_path: Path | None = None) -> int:
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
    processor = build_processor(calibration_path)
    hud = HUDRenderer(settings.ui)
    cap = _open_demo_camera()
    if cap is None:
        logger.warning("Using synthetic demo frames (no usable webcam)")

    consecutive_read_failures = 0

    try:
        processor.initialize()
        display.open()

        frame_idx = 0
        session_start_time = time.time()  # Never reset - used for frame timestamps
        fps_start_time = time.time()  # Reset periodically for FPS calculation
        fps = 0.0
        frame_count = 0

        while True:
            image: NDArray[np.uint8]
            if cap is not None:
                ret, raw_image = cap.read()
                if not ret:
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= 15:
                        logger.warning(
                            "Webcam stopped delivering frames; falling back to synthetic demo"
                        )
                        cap.release()
                        cap = None
                        consecutive_read_failures = 0
                        image = _synthetic_demo_frame()
                    else:
                        # Brief gap — keep the UI alive instead of exiting
                        time.sleep(0.02)
                        action = display.poll_key(wait_ms=10)
                        if action == KeyAction.QUIT or display.was_closed():
                            logger.info("Quit requested during camera warmup/gap")
                            break
                        continue
                else:
                    consecutive_read_failures = 0
                    image = np.asarray(raw_image, dtype=np.uint8)
            else:
                image = _synthetic_demo_frame()

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

            # Handle input (slightly longer wait helps OpenCV event pumping on macOS)
            action = display.poll_key(wait_ms=10)
            if action == KeyAction.QUIT or display.was_closed():
                logger.info("Quit requested")
                break
            elif action == KeyAction.SAVE:
                logger.info("Save requested")
                try:
                    saved_path = save_session_stats(processor.stats)
                    display.show_message(
                        f"Saved {processor.stats.jump_count} jumps -> {saved_path}",
                        duration_ms=2000,
                    )
                except OSError as e:
                    logger.exception("Session save failed")
                    display.show_message(f"Save failed: {e}", duration_ms=2000)

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
        "--calibration",
        type=Path,
        default=None,
        help="Path to calibration profile JSON (default: data/calibration/profile.json)",
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

    if args.demo:
        exit_code = run_demo_mode(calibration_path=args.calibration)
    else:
        exit_code = run_tracking_session(calibration_path=args.calibration)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
