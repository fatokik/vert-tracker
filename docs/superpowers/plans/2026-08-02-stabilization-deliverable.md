# Stabilization Deliverable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a single `vert_tracker` package under uv with RC-based flight safety, calibration bootstrap, and a green test suite.

**Architecture:** Relocate source to `src/vert_tracker/`, migrate Poetry→uv, add `FlightSession` for RC/keepalive/safe shutdown, then wire calibration load/validation and model path.

**Tech Stack:** Python 3.11+, uv, hatchling, pytest, ruff, mypy, djitellopy, mediapipe, OpenCV, pydantic-settings

## Global Constraints

- `requires-python = ">=3.11"`
- Package name `vert_tracker`; CLI entry `vert-tracker = "vert_tracker.main:main"`
- No real drone/camera/display/model download in tests
- Move clamp 20–500 cm; rotate clamp 1–360° (SDK 2.0)
- Keepalive while airborne under 15s SDK auto-land (target ~5s idle)
- Defer detector timing rewrite and Overlay/HUD merge

---

### Task 1: Relocate package + migrate to uv

**Files:**
- Move: `src/{core,analysis,drone,vision,pipeline,ui,main.py}` → `src/vert_tracker/`
- Modify: `pyproject.toml`, `.pre-commit-config.yaml`, `AGENTS.md`, `README.md`
- Modify: all imports in `src/`, `tests/`, `scripts/`
- Create: `uv.lock`
- Delete/stop using: Poetry-only packaging fields; prefer remove `poetry.lock` after `uv.lock` works

**Interfaces:**
- Produces: importable `vert_tracker` package; `uv run vert-tracker`; `uv run pytest`

- [ ] **Step 1: Create feature branch**

```bash
git checkout -b feat/stabilization-uv-package
```

- [ ] **Step 2: Move source tree**

```bash
mkdir -p src/vert_tracker
git mv src/core src/analysis src/drone src/vision src/pipeline src/ui src/main.py src/vert_tracker/
# keep or rewrite src/__init__.py → src/vert_tracker/__init__.py
```

- [ ] **Step 3: Rewrite imports to `vert_tracker.*`**

Update every `from core|analysis|drone|vision|pipeline|ui` import in package, scripts, and tests.

- [ ] **Step 4: Convert pyproject to uv/PEP 621**

Use hatchling, `[project]`, `[project.scripts]`, `[dependency-groups]`, `known-first-party = ["vert_tracker"]`, mypy/pytest paths for `src/vert_tracker`.

- [ ] **Step 5: Lock and sync**

```bash
uv lock && uv sync
```

- [ ] **Step 6: Verify**

```bash
uv run pytest
uv run ruff check .
uv run mypy src
uv run vert-tracker --help
```

Expected: tests collect and pass (existing suite); tools green.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: relocate to vert_tracker package and migrate to uv"
```

---

### Task 2: FlightSession (RC, keepalive, safe shutdown)

**Files:**
- Create: `src/vert_tracker/drone/flight_session.py`
- Modify: `src/vert_tracker/drone/controller.py`
- Modify: `src/vert_tracker/main.py`
- Modify: `src/vert_tracker/drone/stream.py` (avoid double streamon)
- Create: `tests/test_flight_session.py`

**Interfaces:**
- Consumes: `TelloController` with `send_rc`, `takeoff`, `land`, `get_battery`/`cached state`, stream start/stop, disconnect
- Produces:
  - `FlightSession.set_stick(...)` / `apply_key_action(KeyAction)`
  - `FlightSession.takeoff()` / `land()`
  - `FlightSession.battery` (cached)
  - `FlightSession.shutdown_safe()`
  - RC ticker ~10–20 Hz; stick timeout ~0.25s; keepalive ~5s airborne idle

- [ ] **Step 1: Write failing tests** for land-on-shutdown, no-land-when-grounded, RC send while stick set, keepalive, battery not polled every frame.

- [ ] **Step 2: Add controller helpers** — `send_rc`, clamp moves, state-backed battery cache if available.

- [ ] **Step 3: Implement FlightSession**

- [ ] **Step 4: Wire main.py** — replace blocking `move_*` positioning with FlightSession; `finally` → `shutdown_safe()`; remove per-frame `get_battery()`.

- [ ] **Step 5: Fix double streamon**

- [ ] **Step 6: Verify** `uv run pytest tests/test_flight_session.py tests/ -q` and lint/types.

- [ ] **Step 7: Commit** `feat: add FlightSession with RC control and safe shutdown`

---

### Task 3: Calibration bootstrap + model path + docs

**Files:**
- Modify: `src/vert_tracker/vision/calibration.py`, `pose.py`, `pipeline/processor.py`, `main.py`
- Create: `tests/test_calibration_bootstrap.py` (or extend `test_calibration.py`)
- Modify: `README.md`, `AGENTS.md`, `.env.example` if needed

**Interfaces:**
- Produces: `load_calibration_profile(path|default|None)`, validation in Calibrator, CLI `--calibration`, model dir under repo `data/models/`

- [ ] **Step 1: Tests** for valid load, missing→uncalibrated default, invalid px_per_cm reject, model path under `data/models`.

- [ ] **Step 2: Implement validation + bootstrap loader; wire main/processor.**

- [ ] **Step 3: Fix model path helper.**

- [ ] **Step 4: Update README/AGENTS controls and uv commands.**

- [ ] **Step 5: Full verify**

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
uv run vert-tracker --help
```

- [ ] **Step 6: Commit** `feat: load and validate calibration; fix model path`

---
