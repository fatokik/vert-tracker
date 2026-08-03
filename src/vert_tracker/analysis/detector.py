"""Jump detection state machine.

This module is pure logic with NO I/O and NO OpenCV imports.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from vert_tracker.core.config import JumpDetectionSettings
from vert_tracker.core.types import JumpEvent, JumpPhase, Pose


@dataclass
class DetectorState:
    """Internal state for jump detection."""

    phase: JumpPhase = JumpPhase.IDLE
    takeoff_frame: int = 0
    baseline_hip_y: float = 0.0
    peak_hip_y: float = 0.0
    peak_frame: int = 0
    trajectory: list[tuple[int, float]] = field(default_factory=list)
    stable_frames: int = 0


class JumpDetector:
    """State machine for detecting vertical jumps from pose sequences.

    Transitions:
        IDLE → TAKEOFF: Hip velocity exceeds negative threshold (moving up)
        TAKEOFF → AIRBORNE: Continued upward movement confirmed
        AIRBORNE → LANDING: Hip velocity exceeds positive threshold (moving down)
        LANDING → IDLE: Position stabilizes near baseline

    This class is pure logic - no I/O, no OpenCV, no side effects.
    """

    def __init__(self, settings: JumpDetectionSettings | None = None) -> None:
        """Initialize detector with settings.

        Args:
            settings: Jump detection parameters (uses defaults if None)
        """
        self.settings = settings or JumpDetectionSettings()
        self._state = DetectorState()
        self._velocity_buffer: deque[float] = deque(maxlen=5)
        self._position_buffer: deque[float] = deque(maxlen=10)

    @property
    def current_phase(self) -> JumpPhase:
        """Get current jump phase."""
        return self._state.phase

    @property
    def is_jumping(self) -> bool:
        """Check if currently in a jump (not IDLE)."""
        return self._state.phase != JumpPhase.IDLE

    def reset(self) -> None:
        """Reset detector to initial state."""
        self._state = DetectorState()
        self._velocity_buffer.clear()
        self._position_buffer.clear()

    def update(self, pose: Pose) -> JumpEvent | None:
        """Process a new pose and detect jump events.

        Args:
            pose: Current frame's pose data

        Returns:
            JumpEvent if a jump was completed, None otherwise
        """
        hip_center = pose.hip_center
        if hip_center is None:
            return None

        hip_y = hip_center.y
        velocity = self._calculate_velocity(hip_y)

        # Update position buffer for baseline tracking
        self._position_buffer.append(hip_y)

        # State machine transitions
        result = self._process_state(pose.frame_idx, hip_y, velocity)

        return result

    def _calculate_velocity(self, hip_y: float) -> float:
        """Calculate hip vertical velocity.

        Args:
            hip_y: Current hip Y position (normalized)

        Returns:
            Velocity in normalized units per frame (negative = moving up)
        """
        if len(self._velocity_buffer) == 0:
            self._velocity_buffer.append(hip_y)
            return 0.0

        prev_y = self._velocity_buffer[-1]
        velocity = hip_y - prev_y
        self._velocity_buffer.append(hip_y)

        return velocity

    def _process_state(
        self,
        frame_idx: int,
        hip_y: float,
        velocity: float,
    ) -> JumpEvent | None:
        """Process state machine transitions.

        Args:
            frame_idx: Current frame index
            hip_y: Hip Y position (normalized)
            velocity: Hip velocity (normalized units/frame)

        Returns:
            JumpEvent if jump completed, None otherwise
        """
        # Scale velocity to approximate pixels/frame for threshold comparison
        # Assuming ~720p height, multiply by frame height
        scaled_velocity = velocity * 720

        if self._state.phase == JumpPhase.IDLE:
            return self._handle_idle(frame_idx, hip_y, scaled_velocity)

        elif self._state.phase == JumpPhase.TAKEOFF:
            return self._handle_takeoff(frame_idx, hip_y, scaled_velocity)

        elif self._state.phase == JumpPhase.AIRBORNE:
            return self._handle_airborne(frame_idx, hip_y, scaled_velocity)

        elif self._state.phase == JumpPhase.LANDING:
            return self._handle_landing(frame_idx, hip_y, scaled_velocity)

        return None

    def _handle_idle(
        self,
        frame_idx: int,
        hip_y: float,
        velocity: float,
    ) -> JumpEvent | None:
        """Handle IDLE state - watch for takeoff."""
        # Detect upward movement (negative velocity = moving up in image coords)
        if velocity < self.settings.takeoff_velocity_threshold:
            self._state.phase = JumpPhase.TAKEOFF
            self._state.takeoff_frame = frame_idx
            self._state.baseline_hip_y = self._get_baseline_position()
            self._state.peak_hip_y = hip_y
            self._state.peak_frame = frame_idx
            self._state.trajectory = [(frame_idx, hip_y)]

        return None

    def _handle_takeoff(
        self,
        frame_idx: int,
        hip_y: float,
        velocity: float,
    ) -> JumpEvent | None:
        """Handle TAKEOFF state - confirm jump initiation."""
        self._state.trajectory.append((frame_idx, hip_y))

        # Track peak (minimum Y = highest point)
        if hip_y < self._state.peak_hip_y:
            self._state.peak_hip_y = hip_y
            self._state.peak_frame = frame_idx

        # Transition to airborne once we've moved significantly
        frames_since_takeoff = frame_idx - self._state.takeoff_frame
        if frames_since_takeoff >= 2:
            self._state.phase = JumpPhase.AIRBORNE

        # Abort if we're moving down too soon (false trigger)
        if velocity > 0 and frames_since_takeoff < 2:
            self._state.phase = JumpPhase.IDLE
            self._state.trajectory.clear()

        return None

    def _handle_airborne(
        self,
        frame_idx: int,
        hip_y: float,
        velocity: float,
    ) -> JumpEvent | None:
        """Handle AIRBORNE state - track peak and detect landing."""
        self._state.trajectory.append((frame_idx, hip_y))

        # Track peak (minimum Y = highest point)
        if hip_y < self._state.peak_hip_y:
            self._state.peak_hip_y = hip_y
            self._state.peak_frame = frame_idx

        # Check for landing (downward velocity exceeds threshold)
        if velocity > self.settings.landing_velocity_threshold:
            self._state.phase = JumpPhase.LANDING
            self._state.stable_frames = 0

        # Safety check: abort if airborne too long
        airborne_frames = frame_idx - self._state.takeoff_frame
        if airborne_frames > self.settings.max_airborne_frames:
            self._state.phase = JumpPhase.IDLE
            self._state.trajectory.clear()

        return None

    def _handle_landing(
        self,
        frame_idx: int,
        hip_y: float,
        velocity: float,
    ) -> JumpEvent | None:
        """Handle LANDING state - confirm landing and emit event."""
        self._state.trajectory.append((frame_idx, hip_y))

        # Check for position stability near baseline
        baseline_diff = abs(hip_y - self._state.baseline_hip_y)
        velocity_stable = abs(velocity) < 2.0  # Small threshold

        if baseline_diff < 0.05 and velocity_stable:
            self._state.stable_frames += 1
        else:
            self._state.stable_frames = 0

        # Confirm landing after stable frames
        if self._state.stable_frames >= self.settings.landing_stability_frames:
            # Validate jump duration
            airborne_frames = frame_idx - self._state.takeoff_frame
            if airborne_frames >= self.settings.min_airborne_frames:
                event = self._create_jump_event(frame_idx)
                self._state.phase = JumpPhase.IDLE
                self._state.trajectory.clear()
                return event

            # Invalid jump - reset
            self._state.phase = JumpPhase.IDLE
            self._state.trajectory.clear()

        return None

    def _get_baseline_position(self) -> float:
        """Get baseline hip position from recent history."""
        if not self._position_buffer:
            return 0.5  # Default to middle

        # Use median of recent positions
        positions = sorted(self._position_buffer)
        mid = len(positions) // 2
        return positions[mid]

    def _create_jump_event(self, landing_frame: int) -> JumpEvent:
        """Create a JumpEvent from current state.

        Args:
            landing_frame: Frame index of landing

        Returns:
            Completed JumpEvent
        """
        # Height will be calculated by HeightCalculator
        return JumpEvent(
            takeoff_frame=self._state.takeoff_frame,
            peak_frame=self._state.peak_frame,
            landing_frame=landing_frame,
            height_cm=0.0,  # To be calculated
            confidence=self._calculate_confidence(),
            peak_hip_y=self._state.peak_hip_y,
            baseline_hip_y=self._state.baseline_hip_y,
            trajectory=list(self._state.trajectory),
        )

    def _calculate_confidence(self) -> float:
        """Calculate confidence score for the jump detection.

        Returns:
            Confidence score [0, 1]
        """
        # Simple heuristic based on trajectory quality
        trajectory_len = len(self._state.trajectory)

        # More points = higher confidence (up to a point)
        length_score = min(trajectory_len / 30, 1.0)

        # Clear peak = higher confidence
        displacement = self._state.baseline_hip_y - self._state.peak_hip_y
        peak_score = min(displacement / 0.2, 1.0) if displacement > 0 else 0.0

        return (length_score + peak_score) / 2


def detect_jumps_batch(
    poses: list[Pose],
    settings: JumpDetectionSettings | None = None,
) -> list[JumpEvent]:
    """Process a sequence of poses and return all detected jumps.

    Pure function for batch processing recorded data.

    Args:
        poses: Sequence of poses (must be in frame order)
        settings: Detection settings

    Returns:
        List of detected jump events
    """
    detector = JumpDetector(settings)
    events: list[JumpEvent] = []

    for pose in poses:
        event = detector.update(pose)
        if event is not None:
            events.append(event)

    return events
