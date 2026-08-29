# Stabilization Deliverable Design

**Date:** 2026-08-02
**Status:** Draft for review
**Goal:** Make Vert Tracker safely runnable and testable as a single `vert_tracker` package under uv, without adding new product features.

## Problem

The codebase has a coherent layered design, but several integration gaps block trustworthy use:

- Source lives as flat packages under `src/`, while tests/README expect `vert_tracker.*`.
- Poetry is the documented package manager; we want uv.
- Positioning mode can take off, but quit/error cleanup does not guarantee landing.
- Discrete `move_*` commands block the UI loop; Tello auto-lands after 15s without commands.
- Battery is queried every frame instead of using the state stream.
- Saved calibration profiles are not loaded at session start; invalid ratios are not rejected.
- MediaPipe model cache path resolves outside the repo `data/` tree.
- The test suite fails at collection.

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Package layout | `src/vert_tracker/...` |
| Tooling | Migrate Poetry → uv |
| Flight control | RC stick mode (`rc a b c d`) + keepalives |
| Safety scope | Land-on-exit, state-stream telemetry, SDK clamps, RC non-blocking |
| Calibration | Auto-load default/CLI path + validate before use |
| Deliverable scope | Packaging/uv + flight lifecycle + calibration/model path + docs; defer detector timing and HUD merge |
| Implementation order | Big-bang relocate/uv first, then harden flight, then calibration |

## SDK constraints (from README Tello docs)

- Command UDP `8889`, state `8890`, video `11111`, drone IP `192.168.10.1`.
- Move distances: **20–500 cm**.
- Rotation (SDK 2.0): **1–360°** (prefer 2.0 over 1.3’s 1–3600 for EDU).
- **No command for 15 seconds → auto-land.** Airborne sessions need keepalives.
- Prefer state stream fields (`bat`, `h`, …) over per-frame `battery?` reads.
- Discrete `forward`/`left`/… wait for `ok` (blocking). Continuous positioning should use **`rc a b c d`** (−100…100), fire-and-forget.
- Normal shutdown uses **`land`**; **`emergency`** is reserved for panic (optional later).
- SDK 2.0 **`stop`** can force hover.

## Architecture

### Target layout

```
src/vert_tracker/
  __init__.py
  main.py
  core/
  drone/
    controller.py      # thin Tello SDK adapter
    stream.py
    flight_session.py  # NEW: airborne + RC + keepalive + safe shutdown
  vision/
  analysis/
  pipeline/
  ui/
```

### Dependency direction (unchanged)

```
main → ui, drone, pipeline
pipeline → vision, analysis, core
drone → core (+ djitellopy behind adapter)
analysis → core only (no OpenCV/hardware)
```

### FlightSession

New orchestration unit in `vert_tracker.drone.flight_session`.

Responsibilities:

- Track `is_airborne` after successful takeoff / land.
- Maintain RC stick vector from key hold/release edges.
- Run a background RC ticker (~10–20 Hz) calling `TelloController.send_rc(lr, fb, ud, yaw)`.
- On all keys released: send `rc 0 0 0 0`.
- While airborne, if no RC activity for ~5s, send keepalive (`rc 0 0 0 0` or equivalent) so the 15s SDK timeout does not auto-land during tracking.
- Expose cached battery/height from state stream (or throttled reads), never once-per-frame `battery?`.
- `shutdown_safe()`: stop RC ticker → land if airborne → stop stream → disconnect. Attempt land before disconnect; log land failures but still proceed to cleanup.

`main.py` routing:

- Positioning mode: feed key edges into `FlightSession`.
- Tracking mode: no stick motion; keepalives still active while airborne.
- `finally` / quit / unexpected errors: call `shutdown_safe()`.

`TelloController` changes:

- Add `send_rc(...)` wrapper around SDK RC API.
- Clamp discrete move/rotate helpers to SDK 2.0 ranges when retained.
- Avoid duplicate `streamon` (either main or `VideoStream.start`, not both).

### Calibration bootstrap

On session start:

1. If `--calibration PATH` provided → load that profile.
2. Else if `data/calibration/profile.json` exists → load it.
3. Else → settings default `px_per_cm`, mark session **uncalibrated**.

Validation before accepting a profile:

- `px_per_cm` finite and `> 0`
- Known height / marker size `> 0` when that method is used
- Invalid values raise `CalibrationError` (no silent bad ratios)

HUD/status CAL indicator is truthful: OK only for a validated loaded/active profile.

In-app ArUco calibrate (`c`) still updates the active profile; disk save can continue via existing `Calibrator.save_profile` / calibrate script paths.

### Model path

`PoseEstimator` caches under repo `data/models/` (gitignored). Resolve from a stable repo/package root helper, not a brittle multi-`parent` chain that escapes the repo. Keep download-on-first-use.

## Tooling migration (Poetry → uv)

1. Convert `pyproject.toml` to PEP 621 (`[project]`, `[project.scripts]`, `[dependency-groups]`).
2. Prefer `uvx migrate-to-uv` then verify; manual cleanup allowed.
3. Generate `uv.lock`; remove reliance on `poetry.lock` / Poetry-only metadata.
4. Build backend suitable for uv (e.g. hatchling) with:

   ```toml
   [tool.hatch.build.targets.wheel]
   packages = ["src/vert_tracker"]
   ```

   or equivalent `src` layout config so `vert_tracker` installs correctly.

5. Entry point: `vert-tracker = "vert_tracker.main:main"`.
6. `requires-python = ">=3.11"`.
7. Replace docs/scripts/hooks from `poetry run` → `uv run`, `poetry install` → `uv sync`.
8. Update AGENTS.md and README accordingly.

## Testing strategy

All new tests use fakes/mocks. No real takeoff, camera open, model download, or display in CI/unit tests.

Required coverage:

- Imports resolve as `vert_tracker.*`; existing unit tests pass after relocation.
- `FlightSession`: land on shutdown when airborne; no land when grounded; RC ticks while held; keepalive while idle airborne; battery not queried every frame.
- Calibration: load valid; missing → default + uncalibrated; invalid `px_per_cm` rejected.
- Model path resolves under `data/models`.
- Prefer a regression that stream start is not double-invoked if cheap to assert.

Done criteria:

- `uv sync`
- `uv run pytest`
- `uv run ruff check .`
- `uv run mypy src`
- `uv run vert-tracker --help` works; `--demo` starts without import errors (camera/display may still be local-only)

## Documentation updates

- README: uv install/run, package path `src/vert_tracker/`, accurate positioning/tracking controls, Tello links retained.
- AGENTS.md: uv commands, package layout, SDK safety notes (15s keepalive, RC, land-before-disconnect).
- `.env.example`: only if new settings are introduced (RC rate, keepalive interval, calibration path).

## Out of scope

- Jump detector 720p / 30 FPS assumption rewrite
- Merging `OverlayRenderer` and `HUDRenderer`
- Background queue of discrete blocking `move_*` commands
- Hardware CI / claimed ±2–3 cm accuracy verification
- Session save feature completion (unless a trivial existing hook-up)
- Binding an emergency-stop key (optional follow-up)

## Implementation sequence

1. **uv + package relocate** — move modules under `src/vert_tracker/`, fix imports, migrate tooling, green test collection/suite for existing tests.
2. **FlightSession** — RC ticker, keepalive, state-backed battery, land-on-exit, remove per-frame battery spam and blocking positioning moves from the UI loop.
3. **Calibration + model path** — bootstrap load/validate, fix model cache directory, docs/control alignment.

Suggested reviewable PR shape matches the three steps above (or one stabilization PR implemented in that order).

## Risks

- djitellopy RC / state APIs differ slightly by version — wrap behind `TelloController` and mock in tests.
- RC ticker thread must be stopped cleanly to avoid sending after disconnect.
- Keepalive must not fight intentional land.
- uv migration may rewrite dependency pins — verify lock against known-good versions where possible.

## Success definition

Kenny can `uv sync` and `uv run pytest` successfully; the app uses one `vert_tracker` package; an airborne session attempts land before disconnect; positioning uses RC without freezing the frame loop; calibration loads/validates from disk or CLI; no new UI features beyond correcting control docs.
