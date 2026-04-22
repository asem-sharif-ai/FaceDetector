# FaceDetector

A lightweight, drop-in face analysis layer built on top of **MediaPipe Face Mesh**. Wraps the 478-point landmark model with **math and logic** that most vision projects need anyway — Euler angles, EAR, MAR, robust face alignment, environment quality, blendshapes, passive liveness, and gender/age estimation — exposed through a clean, stateless-friendly API.

![demo](demo.gif)

---

## Project Context

This module was originally implemented as the **perception layer** for **[Lynx DMS](https://github.com/asem-sharif-ai/Lynx-DMS)** — an Intelligent Edge Deployed Driver Monitoring System developed as a graduation project at the Faculty of Artificial Intelligence, Menoufia University (Spring 2026).

---

## Motivation

Most face-data pipelines use MediaPipe for detection and landmarks, then re-implement the same handful of geometric methods from scratch every time. This module ships as a standalone package because the problem it solves comes up in essentially every face-aware project. It consolidates that logic once — calibrated thresholds, EMA smoothing, draw modes, PAD engine, ENV engine, GA engine — so you can drop `Detector` into any project and get everything immediately.

---

## Files

| File | What it provides |
|---|---|
| `detector.py` | Core `Detector` class — detection, alignment, geometry, drawing |
| `pad_engine.py` | `PADEngine` — passive anti-spoofing via signal analysis |
| `env_engine.py` | `ENVEngine` — environment quality (brightness, occupancy, jitter) |
| `ga_engine.py` | `GenderAgeEngine` — gender and age estimation via InsightFace ONNX |
| `demo.py` | PyQt6 test harness — camera / video / image input, live metrics |

---

## `Detector` API

### Methods

| Method | Returns |
|---|---|
| `detect(img_bgr)` | landmarks, bbox, blendshapes dict, transformation matrix |
| `align(img_bgr, lm, bbox)` | canonical-pose RGB crop (`size × size`), eye-corner aligned |
| `euler_angles(frame, lm, T)` | yaw, pitch, roll (°) & directional labels |
| `eye_aspect_ratio(frame, lm)` | L-EAR, R-EAR & `Flag` per eye (LOW / NORMAL / HIGH) |
| `mouth_aspect_ratio(frame, lm)` | MAR & `Flag` |
| `draw(img_bgr, lm, bbox)` | annotated RGB frame (`Draw.MODE` / `Draw.COLOR` controlled) |
| `convex_hull(img_bgr, lm)` | binary face mask from face oval landmarks |
| `gradient(img_bgr, lm)` | Sobel gradient magnitude masked to face region |

### Properties

| Property | Returns | Requires |
|---|---|---|
| `env_status` | `(active: bool, result: dict)` | `ACTIVATE_ENV = True` |
| `pad_status` | `(active: bool, result: dict)` | `ACTIVATE_PAD = True` |
| `ga_status` | `(active: bool, gender: str, age: int)` | `ACTIVATE_GA = True` |

All three engines are **opt-in and silent** — disabled by default, zero overhead when off. Enable via class flags before instantiation:

```python
Detector.ACTIVATE_ENV = True
Detector.ACTIVATE_PAD = True
Detector.ACTIVATE_GA  = True
```

### Engine push points

| Engine | Pushed from | Input |
|---|---|---|
| `ENVEngine` | `detect()` | full BGR frame & smoothed landmarks |
| `PADEngine` | `align()` | aligned RGB crop |
| `GAEngine` | `align()` | aligned RGB crop |

---

## Engines

### `PADEngine` — passive anti-spoofing

Scores each aligned face crop across five signal channels, accumulated over a temporal buffer:

- **LBP entropy** — texture richness via Local Binary Patterns
- **HF/LF power ratio** — frequency domain realness check
- **Texture variance** — Gaussian residual sharpness
- **Skin chrominance** — YCbCr Cb/Cr range gate
- **Specular reflection** — highlight ratio plausibility
- **Temporal motion** — inter-frame diff mean & std

`pad_status` returns `(active, {'ready': bool, 'live': bool|None, 'score': float})`. `live=None` means the buffer is not full yet — not a rejection, just undecided.

### `ENVEngine` — environment quality

Buffers brightness, occupancy, and landmark positions over a rolling window and returns stable averaged metrics:

- **Brightness** — mean LAB L-channel over the buffer, normalized 0–1
- **Occupancy** — face bounding area as a fraction of full frame area
- **Jitter** — normalized inter-frame motion across 8 stable landmark points

`env_status` returns `(active, dict)`. Returns `ERROR` flags on all fields if no frames have been pushed yet.

### `GAEngine` — gender & age estimation

Buffers predictions over a rolling window and returns stable results via majority vote and mean:

- **Gender** — majority vote over the buffer (`'Male'` / `'Female'` / `'UnKnown'`)
- **Age** — mean prediction over the buffer, snapped to the nearest 5-year step

Returns `('UnKnown', 0)` until the buffer has enough frames. Requires `genderage.onnx` from InsightFace's buffalo_l model pack — only this single file is needed.

---

## Draw Modes

Controlled via `Draw.MODE` and `Draw.COLOR` at class level (or live via the demo UI dropdowns).

```
BBOX        bounding box only
BBOX_B      bounding box & corner brackets
LANDMARKS   all 478 mesh dots
WIREFRAME   structural skeleton lines
OVERLAY     filled face region (semi-transparent)
OVERLAY_O   overlay & outline
OVERLAY_H   overlay & halo glow
OVERLAY_W   overlay & wireframe
OVERLAY_WO  overlay & wireframe & outline
OVERLAY_WH  overlay & wireframe & halo      ← default
```

+20 named colours available across semantic status colours (`Draw.OK`, `Draw.WARN`, ...) and aesthetic (`Draw.SKY`, `Draw.NEON`, ...).

---

## Quick Start

```python
import cv2
from detector import Detector, Draw

Detector.ACTIVATE_PAD = True   # optional liveness
Detector.ACTIVATE_ENV = True   # optional environment quality
Detector.ACTIVATE_GA  = True   # optional gender/age

detector = Detector(path='detector.task')

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break

    lm, bbox, blendshapes, T = detector.detect(frame)

    if lm is not None:
        y, p, r, ys, ps, rs  = detector.euler_angles(frame, lm, T)
        le, re, les, res     = detector.eye_aspect_ratio(frame, lm)
        mar, ms              = detector.mouth_aspect_ratio(frame, lm)

        _, env               = detector.env_status
        _, pad               = detector.pad_status
        _, gender, age       = detector.ga_status

        drawn   = detector.draw(frame, lm, bbox)    # RGB annotated
        aligned = detector.align(frame, lm, bbox)   # RGB, face crop, size×size

    cv2.imshow('demo', cv2.cvtColor(drawn or frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) == 27: break
```

---

## Demo App

`demo.py` is a **PyQt6 test harness** — not part of the library. Use it to inspect all outputs, tune thresholds, and preview draw modes interactively.

```
python demo.py
```
Supports camera feed, video files, and static images. Live dropdowns control `Draw.MODE` and `Draw.COLOR`. Blendshapes, Euler angles, EAR/MAR, ENV metrics, PAD status, and gender/age are all displayed in real time.

---

## Configuration

Geometic thresholds are defined on `Detector`:

```python
Detector.EULER_ANGLES = (15, 15, 20)   # yaw, pitch, roll (°) before flagging
Detector.EAR_RANGE    = (0.20, 0.40)   # blink / wide-eye bounds
Detector.MAR_RANGE    = (0.05, 0.40)   # closed / open mouth bounds
Detector.YAW_OFFSET   = 0              # initial value of yaw angle
```

ENV thresholds are defined on `ENVEngine`:

```python
ENVEngine.BRIGHTNESS = (0.15, 0.85)    # normalized LAB L-channel bounds
ENVEngine.OCCUPANCY  = (0.02, 0.30)    # face area / frame area
ENVEngine.JITTER     = 0.05            # normalized inter-frame motion
```

### `Detector` constructor

```python
Detector(
    path   = 'detector.task',   # MediaPipe FaceLandmarker model
    size   = 112,               # aligned crop output size
    ratio  = 0.85,              # face-to-crop fill ratio
    smooth = 0.5,               # EMA alpha for landmark smoothing
    buffer = 30,                # PAD / ENV buffer size (GA uses buffer × 2)
)
```

### `Detector._FILTER` — blendshape group filtering

By default cheek and nose groups are suppressed, jaw is filtered to `Open` only. Edit the class-level dict to expose more:

```python
Detector._FILTER = {
    'c': [],        # cheek  — all suppressed
    'n': [],        # nose   — all suppressed
    'j': ['Open']   # jaw    — only jawOpen
}
```

---

## Installation

```bash
git clone https://github.com/asem-sharif-ai/FaceDetector.git
cd FaceDetector
pip install -r requirements.txt
```

---

## Requirements

```
mediapipe
opencv-python
numpy
scikit-image
onnxruntime
PyQt6
```

### Models

| File | How to get |
|---|---|
| `detector.task` | Download manually from [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) |
| `genderage.onnx` | Part of InsightFace [buffalo_l](https://github.com/deepinsight/insightface) — only this file is needed, not the full pack |
