"""Pure analysis logic: jump detection, height calculation, and metrics.

This module contains NO I/O operations and NO OpenCV imports.
All functions operate on typed dataclasses and return results.
"""

from vert_tracker.analysis.calculator import HeightCalculator
from vert_tracker.analysis.detector import JumpDetector
from vert_tracker.analysis.metrics import MetricsTracker

__all__ = ["JumpDetector", "HeightCalculator", "MetricsTracker"]
