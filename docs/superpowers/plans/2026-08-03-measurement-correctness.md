# Measurement Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make jump detection and height metrics trustworthy under variable FPS/resolution, and fix visibility, validate_accuracy, and session-save honesty bugs.

**Architecture:** Drive `JumpDetector` from `pose.timestamp` (norm/sec velocity + second-based airborne gates); require real `frame_height` in the calculator; store timestamps on `JumpEvent` for metrics; then fix the three audit bugs and docs.

**Tech Stack:** Python 3.11+, uv, pytest, existing `vert_tracker` package under `src/vert_tracker/`.

## Global Constraints

- Edit only `src/vert_tracker/` (not orphan paths).
- Keep `analysis` free of OpenCV/hardware/filesystem I/O except existing `export_session_data` JSON write.
- No real drone, camera, model download, or display in unit tests.
- Threshold defaults: takeoff `-0.333` norm/sec, landing `0.333` norm/sec; `min_airborne_s=0.167`, `max_airborne_s=2.0`.
- Landing stability remains consecutive samples; stability velocity ≈ `0.083` norm/sec.
- Overlay/HUD merge is out of scope.
- Spec: `docs/superpowers/specs/2026-08-03-measurement-correctness-design.md`.

---

## File map

| File | Responsibility |
|------|----------------|
| `src/vert_tracker/core/config.py` | Jump settings units/names |
| `src/vert_tracker/core/types.py` | `JumpEvent` timestamps + `airborne_time_s` |
| `src/vert_tracker/analysis/detector.py` | Timestamp velocity + time gates |
| `src/vert_tracker/analysis/calculator.py` | Require positive frame_height; physics uses seconds |
| `src/vert_tracker/analysis/metrics.py` | Airborne time from timestamps |
| `src/vert_tracker/vision/pose.py` | Preserve visibility `0.0` |
| `src/vert_tracker/main.py` | Wire SAVE to export |
| `scripts/validate_accuracy.py` | Capture height before release |
| `tests/conftest.py` | Retune fixtures/settings |
| `tests/test_detector.py` | Timing/dt/airborne tests |
| `tests/test_calculator.py` | frame_height validation |
| `tests/test_metrics.py` (create) | Timestamp airborne time |
| `tests/test_pose_visibility.py` (create) | Visibility zero |
| `tests/test_session_export.py` (create or extend) | Export path |
| `.env.example`, `README.md`, `AGENTS.md` | Units, save path, uv commands |

---

### Task 1: Detector timing + settings + JumpEvent timestamps

**Files:**
- Modify: `src/vert_tracker/core/config.py`
- Modify: `src/vert_tracker/core/types.py`
- Modify: `src/vert_tracker/analysis/detector.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_detector.py`
- Modify: any `JumpEvent(...)` construction sites that break (processor, metrics import, validate_accuracy, calculator tests) — add timestamp kwargs with defaults where needed

**Interfaces:**
- Consumes: `Pose.timestamp`, `Pose.frame_idx`, `Pose.hip_center`
- Produces:
  - `JumpDetectionSettings(takeoff_velocity_threshold: float = -0.333, landing_velocity_threshold: float = 0.333, min_airborne_s: float = 0.167, max_airborne_s: float = 2.0, landing_stability_frames: int = 3)`
  - `JumpEvent(..., takeoff_timestamp: float = 0.0, peak_timestamp: float = 0.0, landing_timestamp: float = 0.0)` with `@property airborne_time_s -> float`
  - `JumpDetector.update(pose) -> JumpEvent | None` using norm/sec velocity

- [ ] **Step 1: Update settings and JumpEvent**

In `config.py` replace jump fields:

```python
class JumpDetectionSettings(BaseSettings):
    """Jump detection algorithm parameters."""

    model_config = SettingsConfigDict(env_prefix="JUMP_")

    takeoff_velocity_threshold: float = -0.333  # normalized units / sec
    landing_velocity_threshold: float = 0.333
    min_airborne_s: float = 0.167
    max_airborne_s: float = 2.0
    landing_stability_frames: int = 3
```

In `types.py` extend `JumpEvent`:

```python
takeoff_frame: int
peak_frame: int
landing_frame: int
height_cm: float
confidence: float
peak_hip_y: float
baseline_hip_y: float
trajectory: list[tuple[int, float]] = field(default_factory=list)
takeoff_timestamp: float = 0.0
peak_timestamp: float = 0.0
landing_timestamp: float = 0.0

@property
def airborne_frames(self) -> int:
    return self.landing_frame - self.takeoff_frame

@property
def airborne_time_s(self) -> float:
    return max(0.0, self.landing_timestamp - self.takeoff_timestamp)
```

- [ ] **Step 2: Write failing detector tests**

Update `tests/conftest.py` fixture:

```python
@pytest.fixture
def jump_detection_settings() -> JumpDetectionSettings:
    return JumpDetectionSettings(
        takeoff_velocity_threshold=-0.333,
        landing_velocity_threshold=0.333,
        min_airborne_s=0.167,
        max_airborne_s=2.0,
        landing_stability_frames=3,
    )
```

Add to `tests/test_detector.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_detector.py -q --no-cov`
Expected: FAIL (missing settings fields / no timestamps / still using 720 scale)

- [ ] **Step 4: Implement detector timestamp velocity and time gates**

Rewrite velocity + state handling in `detector.py`:

```python
_STABILITY_VELOCITY_NORM_PER_S = 0.083  # ~2/720*30 from old scaled units

@dataclass
class DetectorState:
    phase: JumpPhase = JumpPhase.IDLE
    takeoff_frame: int = 0
    takeoff_timestamp: float = 0.0
    baseline_hip_y: float = 0.0
    peak_hip_y: float = 0.0
    peak_frame: int = 0
    peak_timestamp: float = 0.0
    trajectory: list[tuple[int, float]] = field(default_factory=list)
    stable_frames: int = 0

# In JumpDetector.__init__:
self._last_timestamp: float | None = None
self._last_hip_y: float | None = None
# Keep _position_buffer; velocity_buffer of y can go away or store recent velocities

def update(self, pose: Pose) -> JumpEvent | None:
    hip_center = pose.hip_center
    if hip_center is None:
        return None
    hip_y = hip_center.y
    velocity = self._calculate_velocity(hip_y, pose.timestamp)
    if velocity is None:
        # Invalid dt: still update position baseline buffer, skip transitions
        self._position_buffer.append(hip_y)
        return None
    self._position_buffer.append(hip_y)
    return self._process_state(pose.frame_idx, pose.timestamp, hip_y, velocity)

def _calculate_velocity(self, hip_y: float, timestamp: float) -> float | None:
    if self._last_timestamp is None or self._last_hip_y is None:
        self._last_timestamp = timestamp
        self._last_hip_y = hip_y
        return 0.0
    dt = timestamp - self._last_timestamp
    prev_y = self._last_hip_y
    self._last_timestamp = timestamp
    self._last_hip_y = hip_y
    if not math.isfinite(dt) or dt <= 0:
        return None
    return (hip_y - prev_y) / dt
```

Pass `timestamp` through handlers. Replace airborne frame checks:

```python
airborne_s = timestamp - self._state.takeoff_timestamp
if airborne_s > self.settings.max_airborne_s:
    ...
if airborne_s >= self.settings.min_airborne_s:
    event = self._create_jump_event(frame_idx, timestamp)
```

Takeoff confirm may keep **2 consecutive samples** (`frame_idx - takeoff_frame >= 2`) per YAGNI.

Landing stability:

```python
velocity_stable = abs(velocity) < _STABILITY_VELOCITY_NORM_PER_S
```

`_create_jump_event` sets timestamp fields from state + landing args.

`reset()` clears `_last_timestamp` / `_last_hip_y`.

Import `math` at top of detector.

- [ ] **Step 5: Run detector tests**

Run: `uv run pytest tests/test_detector.py -q --no-cov`
Expected: PASS (adjust fixture jump aggressiveness only if 15 FPS case fails after correct thresholds)

- [ ] **Step 6: Commit**

```bash
git add src/vert_tracker/core/config.py src/vert_tracker/core/types.py \
  src/vert_tracker/analysis/detector.py tests/conftest.py tests/test_detector.py \
  src/vert_tracker/pipeline/processor.py src/vert_tracker/analysis/metrics.py \
  scripts/validate_accuracy.py tests/test_calculator.py
git commit -m "$(cat <<'EOF'
feat: drive jump detection from pose timestamps

Use norm/sec velocity and second-based airborne gates; store
takeoff/peak/landing timestamps on JumpEvent.
EOF
)"
```

---

### Task 2: Calculator + metrics (no FPS/height assumptions)

**Files:**
- Modify: `src/vert_tracker/analysis/calculator.py`
- Modify: `src/vert_tracker/analysis/metrics.py`
- Modify: `tests/test_calculator.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `JumpEvent.airborne_time_s`, positive `frame_height`
- Produces:
  - `HeightCalculator.calculate_height(event, frame_height: int) -> float` (no default)
  - `validate_with_physics(height_cm, airborne_time_s: float) -> tuple[bool, float]`
  - `estimate_vertical_velocity(..., frame_height: int, fps: float)` required args, both `> 0`
  - `MetricsTracker.add_jump` uses `event.airborne_time_s`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_calculator.py
def test_rejects_non_positive_frame_height(
    sample_jump_event: JumpEvent,
    calibration_profile: CalibrationProfile,
) -> None:
    calculator = HeightCalculator(calibration_profile)
    with pytest.raises(ValueError):
        calculator.calculate_height(sample_jump_event, frame_height=0)


def test_height_scales_with_frame_height(
    sample_jump_event: JumpEvent,
    calibration_profile: CalibrationProfile,
) -> None:
    calculator = HeightCalculator(calibration_profile)
    h720 = calculator.calculate_height(sample_jump_event, 720)
    h1080 = calculator.calculate_height(sample_jump_event, 1080)
    assert pytest.approx(h1080 / h720, rel=0.01) == 1080 / 720
```

```python
# tests/test_metrics.py
from vert_tracker.analysis.metrics import MetricsTracker
from vert_tracker.core.types import JumpEvent

def test_airborne_time_from_timestamps() -> None:
    event = JumpEvent(
        takeoff_frame=10,
        peak_frame=20,
        landing_frame=40,
        height_cm=40.0,
        confidence=0.9,
        peak_hip_y=0.35,
        baseline_hip_y=0.5,
        takeoff_timestamp=1.0,
        peak_timestamp=1.3,
        landing_timestamp=2.0,
    )
    metrics = MetricsTracker().add_jump(event)
    assert metrics.airborne_time_s == pytest.approx(1.0)
```

Update existing `validate_with_physics` / `estimate_vertical_velocity` call sites in tests to pass required args (no defaults).

- [ ] **Step 2: Run tests to verify fail**

Run: `uv run pytest tests/test_calculator.py tests/test_metrics.py -q --no-cov`
Expected: FAIL on new assertions / old metrics `/ 30.0`

- [ ] **Step 3: Implement calculator + metrics**

```python
def calculate_height(self, event: JumpEvent, frame_height: int) -> float:
    if frame_height <= 0:
        raise ValueError(f"frame_height must be positive, got {frame_height}")
    ...

def validate_with_physics(
    self,
    height_cm: float,
    airborne_time_s: float,
) -> tuple[bool, float]:
    if airborne_time_s <= 0:
        return False, 0.0
    time_to_peak = airborne_time_s / 2
    ...
```

Remove `fps` parameter. Same for module-level helpers: require `frame_height`, require `fps` for `estimate_vertical_velocity` with `if frame_height <= 0 or fps <= 0: raise ValueError`.

```python
# metrics.py
airborne_time = event.airborne_time_s
```

Optionally include timestamps in `export_session_data` JSON jump objects.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_calculator.py tests/test_metrics.py tests/test_detector.py -q --no-cov`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/vert_tracker/analysis/calculator.py src/vert_tracker/analysis/metrics.py \
  tests/test_calculator.py tests/test_metrics.py
git commit -m "$(cat <<'EOF'
feat: remove FPS and 720p assumptions from height metrics

Require positive frame_height and derive airborne time from
JumpEvent timestamps.
EOF
)"
```

---

### Task 3: Bugfixes + docs

**Files:**
- Modify: `src/vert_tracker/vision/pose.py`
- Modify: `scripts/validate_accuracy.py`
- Modify: `src/vert_tracker/main.py`
- Create: `tests/test_pose_visibility.py`
- Create: `tests/test_validate_accuracy_height.py` (small pure helper) OR inline helper in script + test import
- Modify: `.env.example`, `README.md`, `AGENTS.md`

**Interfaces:**
- Consumes: `export_session_data(stats, path)`
- Produces:
  - `_landmark_visibility(lm) -> float` preserving `0.0`
  - `resolve_video_frame_height(cap, first_frame_shape) -> int` used before release
  - `save_session(stats, directory: Path) -> Path` helper in main or metrics

- [ ] **Step 1: Visibility — failing test then fix**

```python
# tests/test_pose_visibility.py
from types import SimpleNamespace
from vert_tracker.vision.pose import landmark_visibility

def test_zero_visibility_preserved() -> None:
    assert landmark_visibility(SimpleNamespace(visibility=0.0)) == 0.0

def test_missing_visibility_defaults_to_one() -> None:
    assert landmark_visibility(SimpleNamespace()) == 1.0
```

Implement in `pose.py`:

```python
def landmark_visibility(lm: object) -> float:
    raw = getattr(lm, "visibility", None)
    if raw is None:
        return 1.0
    value = float(raw)
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))
```

Replace `getattr(...) or 1.0` with `landmark_visibility(lm)`.

- [ ] **Step 2: validate_accuracy — capture height before release**

Extract helper in script (or tiny module) and fix call order:

```python
def resolve_capture_frame_height(cap: cv2.VideoCapture, sample_image: np.ndarray | None) -> int:
    prop = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if prop > 0:
        return prop
    if sample_image is not None and sample_image.ndim >= 2:
        return int(sample_image.shape[0])
    raise ValueError("Could not determine video frame height")
```

In `process_video` (or equivalent): track `first_image` / resolve height **inside** the `try` before `finally: cap.release()`. Use that height for `calculate_height`.

Add a unit test for the helper with a fake cap object.

- [ ] **Step 3: Session save wiring**

```python
# e.g. in main.py or metrics.py
def save_session_stats(stats: SessionStats, directory: Path | None = None) -> Path:
    out_dir = directory or Path("data/sessions")
    path = out_dir / f"session_{time.strftime('%Y%m%d_%H%M%S')}.json"
    export_session_data(stats, path)
    return path
```

In SAVE handler:

```python
elif action == KeyAction.SAVE:
    try:
        path = save_session_stats(processor.stats)
        display.show_message(f"Saved {processor.stats.jump_count} jumps → {path}", duration_ms=2000)
    except Exception as e:
        logger.exception("Session save failed")
        display.show_message(f"Save failed: {e}", duration_ms=2000)
```

Wire demo mode SAVE the same way if it polls keys.

Test export via existing `export_session_data` + `save_session_stats` to `tmp_path`.

- [ ] **Step 4: Docs**

Update `.env.example` JUMP section to norm/sec + `JUMP_MIN_AIRBORNE_S` / `JUMP_MAX_AIRBORNE_S`.

README Development: ensure `uv run pytest` (include pending poetry→uv fix). Document `s` saves to `data/sessions/`.

AGENTS: note timestamp-based detection and session save path.

- [ ] **Step 5: Full verify**

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/vert_tracker/vision/pose.py src/vert_tracker/main.py \
  scripts/validate_accuracy.py tests/test_pose_visibility.py \
  tests/test_validate_accuracy_height.py tests/test_session_export.py \
  .env.example README.md AGENTS.md
git commit -m "$(cat <<'EOF'
fix: preserve visibility zero, capture height before release, save sessions

Wire honest session export and update JUMP_* docs for timestamp-based
detection units.
EOF
)"
```

---

## Self-review checklist

1. **Spec coverage:** Detector timestamps ✓; airborne seconds ✓; calculator frame_height ✓; metrics timestamps ✓; visibility ✓; validate_accuracy ✓; session save ✓; docs ✓; HUD merge deferred ✓
2. **Placeholders:** None intentional; concrete code in each task
3. **Type consistency:** `min_airborne_s` / `max_airborne_s` / `airborne_time_s` / `landmark_visibility` / `save_session_stats` names align across tasks
