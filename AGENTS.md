# AGENTS.md

## Project purpose

Vert Tracker measures vertical jump height from DJI Tello EDU video using
MediaPipe pose estimation, calibration, filtering, and a jump-detection state
machine. It is a Python 3.11+ project managed with Poetry.

The project is still a work in progress. Treat drone control and reported
measurements as safety- and correctness-sensitive.

## Repository map

- `src/main.py` — CLI entry point and application loop.
- `src/core/` — settings, shared data types, exceptions, and logging.
- `src/drone/` — Tello connection, flight commands, and video streaming.
- `src/vision/` — pose estimation, calibration, filtering, and overlays.
- `src/analysis/` — jump detection, height calculation, and session metrics.
- `src/pipeline/processor.py` — per-frame orchestration.
- `src/ui/` — OpenCV display and HUD rendering.
- `tests/` — unit tests for analysis, calibration, and filtering.
- `scripts/` — calibration, recording, and offline validation utilities.
- `data/` — runtime models, calibration profiles, sessions, and recordings;
  most contents are intentionally gitignored.

Source packages currently live directly below `src` (`core`, `analysis`,
`drone`, and so on). The README and tests still contain references to a
`vert_tracker` package that does not exist. Do not introduce a third import
layout; resolve this mismatch consistently when changing packaging.

## Setup and commands

Use Poetry for the project environment:

```bash
poetry install
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
poetry run pre-commit run --all-files
```

Useful application commands:

```bash
poetry run vert-tracker --demo
poetry run vert-tracker
poetry run python scripts/calibrate.py
poetry run python scripts/record_session.py --demo
poetry run python scripts/validate_accuracy.py <video>
```

Prefer demo mode and recorded inputs during development. Running against a
drone requires connecting to its Wi-Fi network and access to real hardware.

## Architecture and data flow

The intended per-frame flow is:

1. `VideoStream` produces a `core.types.Frame`.
2. `PoseEstimator` converts MediaPipe output into project-owned `Pose` and
   `Landmark` types.
3. `LandmarkSmoother` filters landmark coordinates.
4. `JumpDetector` advances the jump state machine and may emit a `JumpEvent`.
5. `HeightCalculator` converts normalized displacement to centimeters using a
   `CalibrationProfile`.
6. `MetricsTracker`, `OverlayRenderer`, and `HUDRenderer` update the session
   output.

Keep `analysis` deterministic and free of hardware, OpenCV, UI, and filesystem
dependencies. Keep hardware/SDK objects behind `drone` and `vision` adapters.
Use the shared types in `core.types` at module boundaries.

## Coding guidelines

- Match Python 3.11 typing and the strict mypy configuration.
- Keep changes small and avoid speculative abstractions.
- Prefer dependency injection for hardware, pose estimation, clocks, and frame
  sources so behavior can be tested without a drone or camera.
- Preserve frame timestamps and dimensions; algorithms must not assume 720p or
  30 FPS unless that constraint is explicitly validated.
- Validate physical inputs such as calibration ratios, distances, heights, and
  FPS before division or issuing SDK commands.
- Do not silently swallow broad exceptions. Catch expected failures, retain
  context, and make fallback behavior observable.
- Avoid duplicating rendering or session state across the pipeline and HUD.
- Update README controls, CLI help, and `.env.example` whenever user-visible
  behavior or configuration changes.

## Testing expectations

- Add regression tests for bug fixes before or with the implementation.
- Unit-test pure analysis logic with synthetic `Pose` and `JumpEvent` data.
- Mock the Tello SDK and OpenCV window functions; normal tests must never issue
  real flight commands, open a camera, download a model, or require a display.
- Add integration tests around `FrameProcessor` with injected fakes rather than
  initializing MediaPipe.
- Test missing poses, non-consecutive frames, variable resolution/FPS, invalid
  calibration, stream startup/cleanup, and state-machine timeouts.
- Run the narrowest relevant tests first, then the full lint/type/test suite.
- If verification cannot run because dependencies or hardware are unavailable,
  report that explicitly; do not claim the change passes.

## Drone safety

- Never take off, move, rotate, or land the drone as part of automated tests.
- Keep takeoff/land state transitions and cleanup paths explicit.
- On shutdown or an unexpected control error, prioritize a safe landing before
  disconnecting when the application knows the drone is airborne.
- Avoid polling battery or sending commands on every video frame; respect SDK
  timing and rate limits.
- Validate movement distances and rotation angles against Tello SDK limits.

## Data and generated files

Do not commit `.env`, recordings, calibration profiles, downloaded model files,
coverage output, or other runtime data. Keep sample fixtures small and
deterministic. Never add credentials or machine-specific absolute paths.

## Before handing off

Summarize changed behavior, list verification commands and outcomes, and call
out anything requiring drone/camera/manual validation. Do not overwrite or
discard unrelated working-tree changes.
