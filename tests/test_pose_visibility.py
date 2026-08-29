"""Tests for landmark visibility extraction."""

from __future__ import annotations

from types import SimpleNamespace

from vert_tracker.vision.pose import landmark_visibility


def test_zero_visibility_preserved() -> None:
    assert landmark_visibility(SimpleNamespace(visibility=0.0)) == 0.0


def test_missing_visibility_defaults_to_one() -> None:
    assert landmark_visibility(SimpleNamespace()) == 1.0


def test_normal_visibility_passthrough() -> None:
    assert landmark_visibility(SimpleNamespace(visibility=0.75)) == 0.75


def test_visibility_clamped_to_valid_range() -> None:
    assert landmark_visibility(SimpleNamespace(visibility=1.5)) == 1.0
    assert landmark_visibility(SimpleNamespace(visibility=-0.5)) == 0.0


def test_non_finite_visibility_treated_as_zero() -> None:
    assert landmark_visibility(SimpleNamespace(visibility=float("nan"))) == 0.0
    assert landmark_visibility(SimpleNamespace(visibility=float("inf"))) == 0.0
