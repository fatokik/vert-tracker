# Measurement Correctness Design

**Date:** 2026-08-03
**Status:** Approved for planning
**Goal:** Make jump detection and height metrics trustworthy under variable FPS/resolution, and fix three known honesty/correctness bugs — without merging Overlay/HUD or adding product features.

## Problem

Phase 1 stabilized packaging, flight safety, and calibration load. Measurement paths still assume ideal capture:

- `JumpDetector` scales velocity by hardcoded `720` and compares thresholds as if at 720p.
- Airborne validity uses frame counts (`min_airborne_frames` / `max_airborne_frames`).
- `MetricsTracker` divides airborne frames by `30.0` FPS.
- Calculator helpers default `frame_height=720` and `fps=30.0`.
- Pose conversion treats landmark `visibility=0.0` as missing via `or 1.0`.
- `validate_accuracy.py` reads `CAP_PROP_FRAME_HEIGHT` after `cap.release()`.
- Main loop claims “Session saved!” while save is a TODO (despite `export_session_data` existing).

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Scope | Timing/frame-size correctness + three audit bugfixes; defer Overlay/HUD merge |
| Velocity | Timestamp-based `dy/dt` in normalized units/sec |
| Airborne gates | Time-based `min_airborne_s` / `max_airborne_s` |
| Session save | Wire `s` to `export_session_data` under `data/sessions/`; success only on write |
| Implementation order | Analysis timing first, then bugfixes |
| Branch | `feat/measurement-correctness` from current stabilization work |

## Out of scope

- Merging `OverlayRenderer` and `HUDRenderer`
- Emergency-stop key
- Hardware CI / claimed ±2–3 cm accuracy verification
- New product features beyond honest session save

## Architecture

### Detector timing

`JumpDetector.update(pose)` uses `pose.timestamp` and hip Y:

- Velocity = `(y - prev_y) / dt` in **normalized units/sec** (negative = up in image coords).
- If `dt <= 0` or non-finite: skip transition (keep phase); do not invent FPS.
- Remove `* 720` scaling entirely. Detector does not take `frame_height`.

**Threshold units:** settings mean norm/sec. Convert current 720p/30fps-equivalent behavior:

- Old compare: `(dy_norm) * 720` vs `-8` → `dy_norm` vs `-8/720` per frame
- At 30 FPS: takeoff ≈ **-0.333 norm/sec**, landing ≈ **+0.333 norm/sec**

**Airborne gates:** replace frame counts with seconds (defaults ≈ `5/30` and `60/30` → **0.167s / 2.0s**). Track takeoff/peak/landing timestamps in detector state; emit them on `JumpEvent`. Keep frame indices for export compatibility.

**Landing stability:**

- `landing_stability_frames` remains consecutive **sample** count (not wall-clock).
- Hardcoded `abs(velocity) < 2.0` (old scaled units) → ~**0.083** norm/sec (`2/720*30`).

### JumpEvent timestamps

Extend `JumpEvent` with `takeoff_timestamp`, `peak_timestamp`, `landing_timestamp` (floats, seconds). Prefer these for airborne duration:

```text
airborne_time_s = landing_timestamp - takeoff_timestamp
```

Trajectory may stay `list[tuple[int, float]]` `(frame_idx, hip_y)` this phase.

### Calculator / metrics

- Public height APIs require positive `frame_height`; reject `<= 0` (no silent `720` default).
- Processor already passes `frame.height` — keep that.
- `validate_with_physics` takes `airborne_time_s` (not frames/fps).
- `MetricsTracker.add_jump` uses JumpEvent timestamps for `airborne_time_s`.
- `estimate_vertical_velocity`: remove default `fps=30` / `frame_height=720`. Callers must pass positive values. Do not expand trajectory to include timestamps this phase unless a production caller needs it (none today).

### Settings / env

`JumpDetectionSettings`:

| Old | New | Default |
|-----|-----|---------|
| `takeoff_velocity_threshold` (-8.0 px-ish) | same name, **norm/sec** | `-0.333` |
| `landing_velocity_threshold` (8.0) | same name, **norm/sec** | `0.333` |
| `min_airborne_frames` (5) | `min_airborne_s` | `0.167` |
| `max_airborne_frames` (60) | `max_airborne_s` | `2.0` |
| `landing_stability_frames` (3) | unchanged (samples) | `3` |

Update `.env.example` comments and key names for airborne seconds.

## Bugfixes

### 1. Visibility zero

In `pose.py`, replace `getattr(lm, "visibility", 1.0) or 1.0` with None-only defaulting so `0.0` is preserved. Clamp to `[0, 1]` if out of range.

### 2. validate_accuracy frame height

Capture frame height while the capture is open (property at start and/or first frame `shape[0]`). Never call `cap.get(...)` after `release()`. Error if height `<= 0` before height calculation.

### 3. Session save

On `KeyAction.SAVE`:

1. Build path `data/sessions/session_<timestamp>.json`.
2. Call `export_session_data(processor.stats, path)`.
3. Success message includes path; failure logs and shows error — never claim success on failure.
4. Empty sessions may still export (honest empty JSON); message can include jump count.
5. Wire the same behavior in demo mode if SAVE is handled there (add if cheap).

## Testing strategy

All unit tests use fakes/synthetics. No drone, camera open, model download, or display.

Required coverage:

- Detector detects existing fixture jump at 30 FPS timestamps after retune.
- Detector still detects (or correctly rejects) under 15 FPS / uneven `dt`.
- `dt <= 0` does not crash or false-trigger.
- Min/max airborne enforced in seconds.
- Calculator rejects `frame_height <= 0`; uses provided height.
- Metrics airborne time from timestamps.
- Pose visibility `0.0` preserved.
- Session export writes JSON; SAVE wiring covered at least at export helper level.
- validate_accuracy height capture order fixed (helper test if extracted).

Done bar:

- `uv run pytest`
- `uv run ruff check .`
- `uv run mypy src`
- No hardcoded `720` / `30` FPS assumptions on detector velocity or metrics airborne-time paths

## Documentation

- `.env.example`: JUMP units → norm/sec and airborne seconds.
- README / AGENTS: timestamp-based detection; session save under `data/sessions/`; uv test commands (include pending README poetry→uv fix if still uncommitted).
- Do not claim hardware accuracy gains; this phase is correctness under variable timing/resolution.

## Implementation sequence

1. **Detector + settings + JumpEvent timestamps** — retune tests/fixtures; green detector suite.
2. **Calculator + metrics** — remove FPS/height defaults on critical paths; physics/airborne time from seconds.
3. **Bugfixes** — visibility, validate_accuracy, session save wiring + docs.

## Risks

- Retuned thresholds may change sensitivity vs old 720p assumption; fixture sequences must be revalidated.
- MediaPipe landmark visibility semantics vary; preserving `0.0` is still correct vs coercing to full visibility.
- OpenCV `CAP_PROP_FRAME_HEIGHT` can be `0` for some backends — prefer first-frame shape fallback.

## Success definition

Kenny can run the test suite green; jump detection uses pose timestamps and second-based gates; height calc requires real frame height; visibility zero is honest; offline validation does not read height after release; pressing `s` writes a real session JSON or reports failure.
