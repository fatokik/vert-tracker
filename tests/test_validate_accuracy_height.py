"""Tests for the validate_accuracy frame-height resolution helper."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from validate_accuracy import resolve_capture_frame_height


def test_prefers_cap_property_when_positive() -> None:
    cap = SimpleNamespace(get=lambda prop: 480)
    sample = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert resolve_capture_frame_height(cap, sample) == 480


def test_falls_back_to_sample_image_when_cap_property_missing() -> None:
    cap = SimpleNamespace(get=lambda prop: 0)
    sample = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert resolve_capture_frame_height(cap, sample) == 720


def test_raises_when_no_height_available() -> None:
    cap = SimpleNamespace(get=lambda prop: 0)
    with pytest.raises(ValueError):
        resolve_capture_frame_height(cap, None)
