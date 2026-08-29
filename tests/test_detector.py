"""Tests for jump detection state machine."""

from __future__ import annotations

from vert_tracker.analysis.detector import JumpDetector, detect_jumps_batch
from vert_tracker.core.config import JumpDetectionSettings
from vert_tracker.core.types import JumpEvent, JumpPhase, Landmark, LandmarkIndex, Pose


def _hip_pose(hip_y: float, timestamp: float, frame_idx: int) -> Pose:
    """Build a minimal pose with hips/ankles at a given height and timestamp."""
    landmarks = {
        LandmarkIndex.LEFT_HIP.value: Landmark(x=0.45, y=hip_y, z=0.0, visibility=0.9),
        LandmarkIndex.RIGHT_HIP.value: Landmark(x=0.55, y=hip_y, z=0.0, visibility=0.9),
        LandmarkIndex.LEFT_ANKLE.value: Landmark(x=0.45, y=hip_y + 0.4, z=0.0, visibility=0.9),
        LandmarkIndex.RIGHT_ANKLE.value: Landmark(x=0.55, y=hip_y + 0.4, z=0.0, visibility=0.9),
    }
    return Pose(landmarks=landmarks, timestamp=timestamp, frame_idx=frame_idx, confidence=0.9)


class TestJumpDetector:
    """Tests for the JumpDetector class."""

    def test_initial_state_is_idle(self, jump_detection_settings: JumpDetectionSettings) -> None:
        """Detector should start in IDLE phase."""
        detector = JumpDetector(jump_detection_settings)
        assert detector.current_phase == JumpPhase.IDLE
        assert not detector.is_jumping

    def test_no_jump_on_standing(
        self,
        standing_pose_sequence: list[Pose],
        jump_detection_settings: JumpDetectionSettings,
    ) -> None:
        """Standing still should not trigger jump detection."""
        detector = JumpDetector(jump_detection_settings)

        for pose in standing_pose_sequence:
            event = detector.update(pose)
            assert event is None

        assert detector.current_phase == JumpPhase.IDLE

    def test_detects_jump_in_sequence(
        self,
        jump_pose_sequence: list[Pose],
        jump_detection_settings: JumpDetectionSettings,
    ) -> None:
        """Should detect a jump in a realistic pose sequence."""
        detector = JumpDetector(jump_detection_settings)
        events = []

        for pose in jump_pose_sequence:
            event = detector.update(pose)
            if event is not None:
                events.append(event)

        # Should detect exactly one jump
        assert len(events) == 1

        # Jump event should have reasonable values
        event = events[0]
        assert event.takeoff_frame > 0
        assert event.peak_frame > event.takeoff_frame
        assert event.landing_frame > event.peak_frame
        assert event.confidence > 0
        # Peak must land at the trajectory's true minimum (frame 22, the
        # parabola vertex), not be misattributed to an earlier frame due to
        # a spurious velocity spike at the takeoff/airborne boundary.
        assert event.peak_frame == 22

    def test_reset_clears_state(
        self,
        jump_pose_sequence: list[Pose],
        jump_detection_settings: JumpDetectionSettings,
    ) -> None:
        """Reset should return detector to initial state."""
        detector = JumpDetector(jump_detection_settings)

        # Process some frames
        for pose in jump_pose_sequence[:20]:
            detector.update(pose)

        # Reset
        detector.reset()

        assert detector.current_phase == JumpPhase.IDLE
        assert not detector.is_jumping

    def test_handles_missing_hip_landmarks(
        self, jump_detection_settings: JumpDetectionSettings
    ) -> None:
        """Should handle poses without hip landmarks gracefully."""
        detector = JumpDetector(jump_detection_settings)

        # Pose without hips
        pose = Pose(
            landmarks={},
            timestamp=0.0,
            frame_idx=0,
            confidence=0.5,
        )

        event = detector.update(pose)
        assert event is None
        assert detector.current_phase == JumpPhase.IDLE


class TestDetectJumpsBatch:
    """Tests for the batch detection function."""

    def test_batch_detection_finds_jumps(
        self,
        jump_pose_sequence: list[Pose],
        jump_detection_settings: JumpDetectionSettings,
    ) -> None:
        """Batch detection should find jumps in sequence."""
        events = detect_jumps_batch(jump_pose_sequence, jump_detection_settings)
        assert len(events) >= 1

    def test_empty_sequence_returns_empty_list(
        self, jump_detection_settings: JumpDetectionSettings
    ) -> None:
        """Empty input should return empty output."""
        events = detect_jumps_batch([], jump_detection_settings)
        assert events == []

    def test_batch_matches_incremental(
        self,
        jump_pose_sequence: list[Pose],
        jump_detection_settings: JumpDetectionSettings,
    ) -> None:
        """Batch and incremental detection should give same results."""
        # Batch detection
        batch_events = detect_jumps_batch(jump_pose_sequence, jump_detection_settings)

        # Incremental detection
        detector = JumpDetector(jump_detection_settings)
        incremental_events = []
        for pose in jump_pose_sequence:
            event = detector.update(pose)
            if event is not None:
                incremental_events.append(event)

        assert len(batch_events) == len(incremental_events)


def test_detects_jump_at_15fps(
    jump_pose_sequence: list[Pose],
    jump_detection_settings: JumpDetectionSettings,
) -> None:
    """Same spatial trajectory at 15 FPS timestamps should still detect."""
    poses = [
        Pose(
            landmarks=p.landmarks,
            timestamp=p.frame_idx / 15.0,
            frame_idx=p.frame_idx,
            confidence=p.confidence,
        )
        for p in jump_pose_sequence
    ]
    events = detect_jumps_batch(poses, jump_detection_settings)
    assert len(events) == 1
    assert events[0].airborne_time_s > 0
    # Peak frame index is spatial (frame_idx-based), independent of the
    # timestamp scale, so it should match the 30 FPS case exactly.
    assert events[0].peak_frame == 22


def test_non_positive_dt_skips_without_crash(
    jump_detection_settings: JumpDetectionSettings,
    sample_pose: Pose,
) -> None:
    detector = JumpDetector(jump_detection_settings)
    p0 = Pose(landmarks=sample_pose.landmarks, timestamp=1.0, frame_idx=0, confidence=0.9)
    p1 = Pose(landmarks=sample_pose.landmarks, timestamp=1.0, frame_idx=1, confidence=0.9)
    assert detector.update(p0) is None
    assert detector.update(p1) is None
    assert detector.current_phase == JumpPhase.IDLE


def test_event_includes_timestamps(
    jump_pose_sequence: list[Pose],
    jump_detection_settings: JumpDetectionSettings,
) -> None:
    events = detect_jumps_batch(jump_pose_sequence, jump_detection_settings)
    assert len(events) == 1
    e = events[0]
    assert e.landing_timestamp > e.takeoff_timestamp
    assert e.peak_timestamp >= e.takeoff_timestamp


def test_uneven_dt_still_detects_or_rejects_without_crash(
    jump_pose_sequence: list[Pose],
    jump_detection_settings: JumpDetectionSettings,
) -> None:
    """Irregular inter-frame spacing must never crash the detector.

    Re-timestamps the same spatial jump trajectory with a non-uniform dt
    pattern (variable frame spacing, as from dropped frames or jitter). The
    detector may either still detect the jump or correctly reject it based
    on the resulting distorted velocities, but it must not raise and must
    settle back into a valid (non-stuck) phase.
    """
    dt_cycle = [0.02, 0.05, 0.01, 0.04, 0.033]
    poses = []
    t = 0.0
    for i, pose in enumerate(jump_pose_sequence):
        if i > 0:
            t += dt_cycle[i % len(dt_cycle)]
        poses.append(
            Pose(
                landmarks=pose.landmarks,
                timestamp=t,
                frame_idx=pose.frame_idx,
                confidence=pose.confidence,
            )
        )

    detector = JumpDetector(jump_detection_settings)
    events: list[JumpEvent] = []
    for pose in poses:
        event = detector.update(pose)
        if event is not None:
            events.append(event)

    # At most one jump should ever be reported, and the long standing tail
    # at the end of the sequence must bring the detector back to IDLE
    # whether or not a jump was ultimately detected.
    assert len(events) <= 1
    assert detector.current_phase == JumpPhase.IDLE


def test_min_airborne_s_rejects_too_short_jump(
    jump_detection_settings: JumpDetectionSettings,
) -> None:
    """A jump that lands well before min_airborne_s must not emit an event."""
    detector = JumpDetector(jump_detection_settings)
    assert jump_detection_settings.min_airborne_s == 0.167

    sequence = [
        (0.5, 0.00, 0),  # baseline, seeds velocity calculation
        (0.0, 0.01, 1),  # sharp upward move -> IDLE -> TAKEOFF
        (-0.4, 0.02, 2),  # continued upward move, still TAKEOFF
        (-0.4, 0.03, 3),  # 2-frame confirmation -> TAKEOFF -> AIRBORNE
        (0.5, 0.04, 4),  # sharp downward move -> AIRBORNE -> LANDING
        (0.5, 0.05, 5),  # stable near baseline (1/3)
        (0.5, 0.06, 6),  # stable near baseline (2/3)
        (0.5, 0.07, 7),  # stable near baseline (3/3) -> confirmation check
    ]

    events: list[JumpEvent] = []
    for hip_y, timestamp, frame_idx in sequence:
        event = detector.update(_hip_pose(hip_y, timestamp, frame_idx))
        if event is not None:
            events.append(event)

    # Total airborne time (0.03s) is far below min_airborne_s (0.167s), so
    # the jump must be rejected rather than emitted, and the detector must
    # reset cleanly back to IDLE.
    assert events == []
    assert detector.current_phase == JumpPhase.IDLE


def test_max_airborne_s_aborts_without_emitting_jump(
    jump_detection_settings: JumpDetectionSettings,
) -> None:
    """A jump that stays airborne past max_airborne_s must abort, not emit."""
    detector = JumpDetector(jump_detection_settings)
    assert jump_detection_settings.max_airborne_s == 2.0

    sequence = [
        (0.5, 0.0, 0),  # baseline
        (0.0, 0.1, 1),  # sharp upward move -> IDLE -> TAKEOFF
        (-0.4, 0.2, 2),  # continued upward move, still TAKEOFF
        (-0.4, 0.3, 3),  # 2-frame confirmation -> TAKEOFF -> AIRBORNE
        (-0.4, 1.4, 4),  # holds position (no landing velocity), 1.1s elapsed
        (-0.4, 2.5, 5),  # holds position; total airborne time now 2.4s
    ]

    events: list[JumpEvent] = []
    for hip_y, timestamp, frame_idx in sequence:
        event = detector.update(_hip_pose(hip_y, timestamp, frame_idx))
        if event is not None:
            events.append(event)

    # Never crosses the landing velocity threshold, so the safety timeout
    # (not a landing) must be what returns the detector to IDLE.
    assert events == []
    assert detector.current_phase == JumpPhase.IDLE
