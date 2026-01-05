# Vert Tracker

Measure vertical jump height for volleyball training using a DJI Tello EDU drone and computer vision.

## Features

- **Real-time measurement** - Jump height calculated and displayed live during training
- **Pose tracking** - MediaPipe 33-landmark body tracking for accurate motion capture
- **Reference calibration** - ArUco marker or known-object calibration for pixel-to-cm conversion
- **Session history** - Track jump metrics over time with statistics
- **Visual feedback** - Skeleton overlay, trajectory visualization, and HUD metrics

## Hardware Requirements

- DJI Tello EDU drone
- Laptop with WiFi capability
- (Optional) Printed ArUco marker for calibration

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vert-tracker.git
cd vert-tracker

# Install dependencies with Poetry
poetry install

# Set up pre-commit hooks
poetry run pre-commit install

# Copy and configure environment
cp .env.example .env
```

## Quick Start

```bash
# 1. Connect laptop to Tello WiFi network (TELLO-XXXXXX)

# 2. Run calibration (first time or when changing setup)
poetry run python scripts/calibrate.py

# 3. Start tracking session
poetry run vert-tracker
```

## Controls

- `q` - Quit application
- `c` - Run calibration
- `r` - Reset session statistics
- `s` - Save current session
- `Space` - Pause/resume tracking

## Project Structure

```
vert-tracker/
├── src/vert_tracker/
│   ├── core/           # Config, types, exceptions, logging
│   ├── drone/          # Tello control and video streaming
│   ├── vision/         # Pose estimation, calibration, filters
│   ├── analysis/       # Jump detection and height calculation (pure logic)
│   ├── pipeline/       # Frame processing orchestration
│   └── ui/             # Display and HUD rendering
├── tests/              # Unit tests
├── scripts/            # Standalone utilities
└── data/               # Calibration, sessions, recordings
```

## Development Milestones

### M1: Tello Connection + Video Stream
- [ ] Connect to Tello EDU
- [ ] Stream 720p@30fps video to OpenCV window
- [ ] Basic takeoff/land/hover controls

### M2: MediaPipe Pose Overlay
- [ ] Integrate MediaPipe pose estimation
- [ ] Draw skeleton overlay on video feed
- [ ] Track hip center as primary reference point

### M3: Calibration System
- [ ] ArUco marker detection
- [ ] Known-height reference object calibration
- [ ] Persist calibration profiles

### M4: Jump Detection Algorithm
- [ ] State machine: IDLE → TAKEOFF → AIRBORNE → LANDING
- [ ] Velocity-based phase transitions
- [ ] Robust handling of noise and partial poses

### M5: Height Calculation with Validation
- [ ] Peak detection via trajectory fitting
- [ ] Pixel displacement to cm conversion
- [ ] Cross-validation with physics model

### M6: Real-time UI with History
- [ ] Live HUD with current jump metrics
- [ ] Session statistics display
- [ ] Jump history timeline

### M7: Polish and Robustness
- [ ] Edge case handling
- [ ] Performance optimization
- [ ] Comprehensive testing

## Technical Approach

### Computer Vision Pipeline

1. **Frame Capture** - Tello streams H.264 video, decoded by OpenCV
2. **Pose Estimation** - MediaPipe extracts 33 body landmarks per frame
3. **Filtering** - Kalman filter smooths landmark positions, reduces jitter
4. **Phase Detection** - State machine tracks jump phases via hip velocity
5. **Height Calculation** - Peak hip displacement converted to cm using calibration
6. **Visualization** - Skeleton overlay, trajectory, and metrics rendered on frame

### Calibration

The system uses a reference object of known size (ArUco marker or athlete height) to establish the pixels-per-centimeter ratio at the capture distance. This ratio is used to convert pixel displacement to real-world height.

### Jump Detection State Machine

```
IDLE ──(velocity < -threshold)──> TAKEOFF
TAKEOFF ──(feet leave ground)──> AIRBORNE
AIRBORNE ──(velocity > threshold)──> LANDING
LANDING ──(stable on ground)──> IDLE
```

### Target Accuracy

The system targets ±2-3 cm accuracy under optimal conditions:
- Good lighting
- Clear side view
- Stable drone position
- Proper calibration

## Development

```bash
# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov

# Lint and format
poetry run ruff check .
poetry run ruff format .

# Type checking
poetry run mypy src
```

## License

MIT License - see LICENSE file for details.
