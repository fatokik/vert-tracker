"""Session statistics and metrics tracking.

This module is pure logic with NO I/O and NO OpenCV imports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vert_tracker.core.types import JumpEvent, SessionStats


@dataclass
class JumpMetrics:
    """Computed metrics for a single jump."""

    height_cm: float
    airborne_time_s: float
    peak_velocity_estimate: float
    confidence: float


@dataclass
class SessionSummary:
    """Summary statistics for a training session."""

    total_jumps: int
    max_height_cm: float | None
    avg_height_cm: float | None
    std_height_cm: float | None
    min_height_cm: float | None
    total_duration_s: float
    jumps_per_minute: float


class MetricsTracker:
    """Tracks and computes session metrics.

    Maintains a SessionStats object and provides computed statistics.
    """

    def __init__(self, stats: SessionStats | None = None) -> None:
        """Initialize tracker with optional existing stats.

        Args:
            stats: Existing session stats to continue tracking
        """
        self.stats = stats or SessionStats()

    @property
    def jump_count(self) -> int:
        """Get total jump count."""
        return self.stats.jump_count

    @property
    def max_height(self) -> float | None:
        """Get maximum jump height."""
        return self.stats.max_height

    @property
    def avg_height(self) -> float | None:
        """Get average jump height."""
        return self.stats.avg_height

    @property
    def last_jump(self) -> JumpEvent | None:
        """Get most recent jump."""
        return self.stats.last_jump

    def add_jump(self, event: JumpEvent) -> JumpMetrics:
        """Add a jump event and compute metrics.

        Args:
            event: Completed jump event

        Returns:
            Computed metrics for the jump
        """
        self.stats.add_jump(event)

        # Compute metrics
        airborne_time = event.airborne_frames / 30.0  # Assuming 30fps

        # Estimate peak velocity from displacement and time
        # v = sqrt(2 * g * h)
        g = 980.0  # cm/s^2
        peak_velocity = (2 * g * event.height_cm) ** 0.5 if event.height_cm > 0 else 0.0

        return JumpMetrics(
            height_cm=event.height_cm,
            airborne_time_s=airborne_time,
            peak_velocity_estimate=peak_velocity,
            confidence=event.confidence,
        )

    def get_summary(self, session_duration_s: float = 0.0) -> SessionSummary:
        """Get session summary statistics.

        Args:
            session_duration_s: Total session duration in seconds

        Returns:
            SessionSummary with computed statistics
        """
        heights = [j.height_cm for j in self.stats.jumps]

        min_height = min(heights) if heights else None
        jumps_per_min = (
            self.jump_count / (session_duration_s / 60.0) if session_duration_s > 0 else 0.0
        )

        return SessionSummary(
            total_jumps=self.jump_count,
            max_height_cm=self.max_height,
            avg_height_cm=self.avg_height,
            std_height_cm=self.stats.std_height,
            min_height_cm=min_height,
            total_duration_s=session_duration_s,
            jumps_per_minute=jumps_per_min,
        )

    def get_recent_jumps(self, count: int = 5) -> list[JumpEvent]:
        """Get most recent jumps.

        Args:
            count: Number of recent jumps to return

        Returns:
            List of recent jump events (newest first)
        """
        return list(reversed(self.stats.jumps[-count:]))

    def get_height_trend(self) -> list[float]:
        """Get sequence of jump heights for trend analysis.

        Returns:
            List of heights in chronological order
        """
        return [j.height_cm for j in self.stats.jumps]

    def get_improvement(self, window: int = 5) -> float | None:
        """Calculate improvement between early and recent jumps.

        Args:
            window: Number of jumps to compare at start/end

        Returns:
            Percentage improvement or None if insufficient data
        """
        if len(self.stats.jumps) < window * 2:
            return None

        early_avg = sum(j.height_cm for j in self.stats.jumps[:window]) / window
        recent_avg = sum(j.height_cm for j in self.stats.jumps[-window:]) / window

        if early_avg == 0:
            return None

        return ((recent_avg - early_avg) / early_avg) * 100

    def reset(self) -> None:
        """Clear all recorded data."""
        self.stats.reset()


def compute_percentile(heights: list[float], percentile: float) -> float | None:
    """Compute percentile of jump heights.

    Args:
        heights: List of jump heights
        percentile: Percentile to compute (0-100)

    Returns:
        Percentile value or None if empty list
    """
    if not heights:
        return None

    sorted_heights = sorted(heights)
    idx = int((percentile / 100.0) * (len(sorted_heights) - 1))
    return sorted_heights[idx]


def analyze_consistency(heights: list[float]) -> dict[str, float | None]:
    """Analyze jump height consistency.

    Args:
        heights: List of jump heights

    Returns:
        Dictionary with consistency metrics
    """
    if len(heights) < 2:
        return {
            "coefficient_of_variation": None,
            "range": None,
            "interquartile_range": None,
        }

    mean = sum(heights) / len(heights)
    std = (sum((h - mean) ** 2 for h in heights) / len(heights)) ** 0.5

    cv = (std / mean * 100) if mean > 0 else None
    height_range = max(heights) - min(heights)

    q25 = compute_percentile(heights, 25)
    q75 = compute_percentile(heights, 75)
    iqr = (q75 - q25) if q25 is not None and q75 is not None else None

    return {
        "coefficient_of_variation": cv,
        "range": height_range,
        "interquartile_range": iqr,
    }


def export_session_data(
    stats: SessionStats,
    path: Path,
) -> None:
    """Export session data to JSON file.

    Args:
        stats: Session statistics to export
        path: Output file path
    """
    jumps_data = [
        {
            "takeoff_frame": j.takeoff_frame,
            "peak_frame": j.peak_frame,
            "landing_frame": j.landing_frame,
            "height_cm": j.height_cm,
            "confidence": j.confidence,
            "airborne_frames": j.airborne_frames,
        }
        for j in stats.jumps
    ]

    data = {
        "start_time": stats.start_time,
        "jump_count": stats.jump_count,
        "max_height": stats.max_height,
        "avg_height": stats.avg_height,
        "std_height": stats.std_height,
        "jumps": jumps_data,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def import_session_data(path: Path) -> SessionStats:
    """Import session data from JSON file.

    Args:
        path: Input file path

    Returns:
        Reconstructed SessionStats
    """
    with open(path) as f:
        data = json.load(f)

    stats = SessionStats(start_time=data.get("start_time", 0.0))

    for j in data.get("jumps", []):
        event = JumpEvent(
            takeoff_frame=j["takeoff_frame"],
            peak_frame=j["peak_frame"],
            landing_frame=j["landing_frame"],
            height_cm=j["height_cm"],
            confidence=j["confidence"],
            peak_hip_y=0.0,
            baseline_hip_y=0.0,
        )
        stats.add_jump(event)

    return stats
