#!/usr/bin/env python3
"""Validate jump height measurement accuracy.

Process recorded videos and compare measured heights against
known reference values for accuracy assessment.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vert_tracker.analysis.calculator import HeightCalculator
from vert_tracker.analysis.detector import detect_jumps_batch
from vert_tracker.core.config import get_settings
from vert_tracker.core.logging import get_logger, setup_logging
from vert_tracker.core.types import CalibrationProfile, Frame, JumpEvent, Pose
from vert_tracker.vision.calibration import Calibrator
from vert_tracker.vision.pose import PoseEstimator

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single jump."""

    jump_index: int
    measured_height_cm: float
    reference_height_cm: float | None
    error_cm: float | None
    error_percent: float | None


@dataclass
class ValidationSummary:
    """Summary statistics for validation run."""

    total_jumps_detected: int
    total_jumps_reference: int
    matched_jumps: int
    mean_absolute_error_cm: float | None
    std_error_cm: float | None
    max_error_cm: float | None
    mean_error_percent: float | None


def process_video(
    video_path: Path,
    calibration: CalibrationProfile,
    fps: float = 30.0,
) -> tuple[list[JumpEvent], list[Pose]]:
    """Process a video file and detect jumps.

    Args:
        video_path: Path to video file
        calibration: Calibration profile for height calculation
        fps: Video frame rate

    Returns:
        Tuple of (jump_events, all_poses)
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    pose_estimator = PoseEstimator()
    pose_estimator.initialize()

    poses: list[Pose] = []
    frame_idx = 0

    logger.info("Processing video: %s", video_path)

    try:
        while True:
            ret, image = cap.read()
            if not ret:
                break

            frame = Frame(
                image=image,
                timestamp=frame_idx / fps,
                index=frame_idx,
            )

            pose = pose_estimator.estimate(frame)
            if pose is not None:
                poses.append(pose)

            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.info("Processed %d frames...", frame_idx)

    finally:
        cap.release()
        pose_estimator.close()

    logger.info("Processed %d frames, detected poses in %d", frame_idx, len(poses))

    # Detect jumps
    events = detect_jumps_batch(poses)

    # Calculate heights
    calculator = HeightCalculator(calibration)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    calculated_events = []
    for event in events:
        jump_height = calculator.calculate_height(event, height)
        calculated_events.append(
            JumpEvent(
                takeoff_frame=event.takeoff_frame,
                peak_frame=event.peak_frame,
                landing_frame=event.landing_frame,
                height_cm=jump_height,
                confidence=event.confidence,
                peak_hip_y=event.peak_hip_y,
                baseline_hip_y=event.baseline_hip_y,
                trajectory=event.trajectory,
            )
        )

    logger.info("Detected %d jumps", len(calculated_events))

    return calculated_events, poses


def load_reference_data(csv_path: Path) -> list[tuple[int, float]]:
    """Load reference jump heights from CSV.

    Expected format: frame_number,height_cm

    Args:
        csv_path: Path to CSV file

    Returns:
        List of (frame_index, height_cm) tuples
    """
    references = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row.get("frame", row.get("frame_number", 0)))
            height = float(row.get("height_cm", row.get("height", 0)))
            references.append((frame, height))

    logger.info("Loaded %d reference measurements", len(references))
    return references


def match_jumps_to_references(
    jumps: list[JumpEvent],
    references: list[tuple[int, float]],
    frame_tolerance: int = 30,
) -> list[ValidationResult]:
    """Match detected jumps to reference measurements.

    Args:
        jumps: Detected jump events
        references: Reference (frame, height) pairs
        frame_tolerance: Frame window for matching

    Returns:
        List of validation results
    """
    results = []

    for i, jump in enumerate(jumps):
        # Find closest reference by peak frame
        best_match = None
        best_distance = float("inf")

        for ref_frame, ref_height in references:
            distance = abs(jump.peak_frame - ref_frame)
            if distance < best_distance and distance <= frame_tolerance:
                best_distance = distance
                best_match = (ref_frame, ref_height)

        if best_match is not None:
            ref_height = best_match[1]
            error = jump.height_cm - ref_height
            error_pct = (error / ref_height * 100) if ref_height > 0 else None

            results.append(
                ValidationResult(
                    jump_index=i,
                    measured_height_cm=jump.height_cm,
                    reference_height_cm=ref_height,
                    error_cm=error,
                    error_percent=error_pct,
                )
            )
        else:
            results.append(
                ValidationResult(
                    jump_index=i,
                    measured_height_cm=jump.height_cm,
                    reference_height_cm=None,
                    error_cm=None,
                    error_percent=None,
                )
            )

    return results


def compute_summary(
    results: list[ValidationResult],
    total_references: int,
) -> ValidationSummary:
    """Compute summary statistics from validation results.

    Args:
        results: Individual validation results
        total_references: Total number of reference measurements

    Returns:
        ValidationSummary with statistics
    """
    matched = [r for r in results if r.error_cm is not None]

    if not matched:
        return ValidationSummary(
            total_jumps_detected=len(results),
            total_jumps_reference=total_references,
            matched_jumps=0,
            mean_absolute_error_cm=None,
            std_error_cm=None,
            max_error_cm=None,
            mean_error_percent=None,
        )

    errors = [abs(r.error_cm) for r in matched if r.error_cm is not None]
    error_pcts = [abs(r.error_percent) for r in matched if r.error_percent is not None]

    return ValidationSummary(
        total_jumps_detected=len(results),
        total_jumps_reference=total_references,
        matched_jumps=len(matched),
        mean_absolute_error_cm=np.mean(errors),
        std_error_cm=np.std(errors),
        max_error_cm=max(errors),
        mean_error_percent=np.mean(error_pcts) if error_pcts else None,
    )


def print_results(
    results: list[ValidationResult],
    summary: ValidationSummary,
) -> None:
    """Print validation results to console."""
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print("\nIndividual Jumps:")
    print("-" * 60)
    print(f"{'Jump':<6} {'Measured':<12} {'Reference':<12} {'Error':<12} {'Error %':<10}")
    print("-" * 60)

    for r in results:
        ref_str = f"{r.reference_height_cm:.1f}" if r.reference_height_cm else "N/A"
        err_str = f"{r.error_cm:+.1f}" if r.error_cm is not None else "N/A"
        pct_str = f"{r.error_percent:+.1f}%" if r.error_percent is not None else "N/A"

        print(
            f"{r.jump_index:<6} "
            f"{r.measured_height_cm:<12.1f} "
            f"{ref_str:<12} "
            f"{err_str:<12} "
            f"{pct_str:<10}"
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Jumps detected:      {summary.total_jumps_detected}")
    print(f"Reference jumps:     {summary.total_jumps_reference}")
    print(f"Matched jumps:       {summary.matched_jumps}")

    if summary.mean_absolute_error_cm is not None:
        print(f"\nMean Absolute Error: {summary.mean_absolute_error_cm:.2f} cm")
        print(f"Std Dev Error:       {summary.std_error_cm:.2f} cm")
        print(f"Max Error:           {summary.max_error_cm:.2f} cm")

        if summary.mean_error_percent is not None:
            print(f"Mean Error %:        {summary.mean_error_percent:.1f}%")

        # Target accuracy check
        target = 3.0  # cm
        if summary.mean_absolute_error_cm <= target:
            print(
                f"\n✓ PASS: Mean error ({summary.mean_absolute_error_cm:.2f} cm) "
                f"<= target ({target} cm)"
            )
        else:
            print(
                f"\n✗ FAIL: Mean error ({summary.mean_absolute_error_cm:.2f} cm) "
                f"> target ({target} cm)"
            )


def main() -> int:
    """Run validation script."""
    parser = argparse.ArgumentParser(description="Validate jump height measurement accuracy")
    parser.add_argument(
        "video",
        type=Path,
        help="Path to recorded video file",
    )
    parser.add_argument(
        "--reference",
        "-r",
        type=Path,
        help="Path to CSV with reference measurements",
    )
    parser.add_argument(
        "--calibration",
        "-c",
        type=Path,
        help="Path to calibration profile JSON",
    )
    parser.add_argument(
        "--px-per-cm",
        type=float,
        default=5.0,
        help="Manual px/cm value if no calibration file",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video frame rate (default: 30)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output CSV for results",
    )

    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.logging.level)

    # Load calibration
    if args.calibration and args.calibration.exists():
        calibrator = Calibrator()
        calibration = calibrator.load_profile(args.calibration)
    else:
        from vert_tracker.core.types import CalibrationMethod

        calibration = CalibrationProfile(
            px_per_cm=args.px_per_cm,
            method=CalibrationMethod.MANUAL,
            distance_cm=250.0,
            timestamp=0.0,
        )

    # Process video
    jumps, poses = process_video(args.video, calibration, args.fps)

    if not jumps:
        logger.warning("No jumps detected in video")
        return 1

    # Load references if provided
    if args.reference and args.reference.exists():
        references = load_reference_data(args.reference)
        results = match_jumps_to_references(jumps, references)
        summary = compute_summary(results, len(references))
    else:
        # No reference - just show detected jumps
        results = [
            ValidationResult(
                jump_index=i,
                measured_height_cm=j.height_cm,
                reference_height_cm=None,
                error_cm=None,
                error_percent=None,
            )
            for i, j in enumerate(jumps)
        ]
        summary = ValidationSummary(
            total_jumps_detected=len(jumps),
            total_jumps_reference=0,
            matched_jumps=0,
            mean_absolute_error_cm=None,
            std_error_cm=None,
            max_error_cm=None,
            mean_error_percent=None,
        )

    print_results(results, summary)

    # Save results if output specified
    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "jump_index",
                    "measured_cm",
                    "reference_cm",
                    "error_cm",
                    "error_percent",
                ]
            )
            for r in results:
                writer.writerow(
                    [
                        r.jump_index,
                        r.measured_height_cm,
                        r.reference_height_cm or "",
                        r.error_cm or "",
                        r.error_percent or "",
                    ]
                )
        logger.info("Results saved to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
