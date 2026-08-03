"""Tests for jump detection state machine."""

from __future__ import annotations

from vert_tracker.analysis.detector import JumpDetector, detect_jumps_batch
from vert_tracker.core.config import JumpDetectionSettings
from vert_tracker.core.types import JumpPhase, Pose


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
