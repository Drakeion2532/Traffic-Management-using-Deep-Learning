# AI-Powered Traffic Optimization with YOLOv8

[![tests](https://github.com/Drakeion2532/Traffic-Management-using-Deep-Learning/actions/workflows/tests.yml/badge.svg)](https://github.com/Drakeion2532/Traffic-Management-using-Deep-Learning/actions/workflows/tests.yml)

Real-time vehicle detection, tracking, lane-wise counting and **adaptive traffic-signal timing** from ordinary CCTV footage.

This is the reference implementation of the paper
**["Real-Time Traffic Management Using Deep Learning and Object Detection using YOLOv8"](https://ieeexplore.ieee.org/document/10725412)** (IEEE ICCCNT 2024) by
S. Phatangare, R. Sakpal, S. Khan, **S. Punde**, S. Kasurde (Vishwakarma Institute of Technology, Pune).

![demo](docs/demo.gif)

## The problem

Indian junctions mostly run **fixed-time signals**: every approach gets the same green time whether it has 40 vehicles queued or none. That wastes green time on empty roads while the busy approaches stay jammed. Pressure mats and loop detectors are expensive and wear out, but most junctions already have CCTV cameras.

## What this system does

```
CCTV / video ──► YOLOv8 detection ──► ByteTrack IDs ──► line-crossing counts per lane
                                                    └─► live density (PCU) in ROI
                                                                  │
                                       Adaptive Signal Timer ◄────┘
                                (per-class clear time, lanes, start-up lag,
                                 min/max green to prevent starvation)
```

| Module | File | Purpose |
|---|---|---|
| Detection + tracking | `traffic_ai/video_pipeline.py` | YOLOv8 (car, motorbike, bus, truck, bicycle), ByteTrack IDs, ROI filtering, lane-wise counting, annotated video + CSV log |
| Signal timing | `traffic_ai/signal_timer.py` | Green time = ⌈Σ(countᶜ × clear-timeᶜ) / (lanes + 1)⌉ + start-up lag, clamped to [10 s, 60 s]. The signal order stays fixed, as in the paper. Only the durations adapt. |
| Evaluation | `traffic_ai/simulation.py` | 4-way junction simulator comparing static vs adaptive control (12 runs × 5 min, as in the paper) |

## Results

### 1. Static vs adaptive control (simulation)

`python -m traffic_ai.simulation`: 12 simulations × 5 minutes with unbalanced demand on the four approaches. The adaptive controller only sees what the camera sees: each queued vehicle is detected with 90% recall.

| Metric | Static 30 s timer | Adaptive (YOLO counts) |
|---|---|---|
| Vehicles crossed (12 runs) | 2,114 | **2,412 (+14.1%)** |
| Avg. idle green per run | 87.2 s | **25.0 s (−71%)** |
| Avg. wait per vehicle | 64.8 s | 62.6 s (−3%) |

![simulation](docs/simulation_results.png)

The roughly 14% throughput gain matches the ≈15% improvement over static control reported in the paper's simulations. Average wait barely moves (−3%): the adaptive timer mainly stops wasting green on empty approaches, while queues on the busiest approach still build up. Changing the phase *order* by demand, not just the durations, helps further; see 1b.

### 1b. Counting every vehicle, and a smarter phase order (beyond the paper)

The paper's wait metric only counts vehicles that got through. Cars still stuck in the queue when the 5 minutes end are left out, and those are exactly the worst waits. Counting **every** vehicle (`avg_delay_all_s`) is the fairer number for commuters, and it shows the camera-based timer helps far more than the 3% above:

| Over 12 runs | Static 30 s | Adaptive (paper) | **Demand-ordered** |
|---|---|---|---|
| Vehicles crossed | 2,114 | 2,412 (+14.1%) | **2,450 (+15.9%)** |
| Avg. delay, every vehicle | 68.5 s | 59.2 s (−14%) | **54.3 s (−21%)** |
| Left queued at the end (avg/run) | 139 | 114 | **111** |
| Longest red with cars waiting | 99 s | 118 s | 114 s |

**Demand-ordered** (`--controller demand` in `simulate`) keeps the paper's camera-based green durations, but it also chooses **which approach goes next**:

* **Busiest first:** the next green goes to the approach with the most detected traffic (in PCU), and empty approaches are skipped.
* **No starvation:** no approach with waiting vehicles is meant to sit on red for more than 90 s. The current green is cut short (after the 10 s minimum) when someone is about to pass that limit. When two approaches become overdue together, the second still waits a little longer, so the worst case observed was 114 s.

The paper keeps a fixed phase order, which some junctions need for driver expectations or pedestrian phases, so its controller is unchanged and its numbers above are reproduced exactly.

### 2. Real video

`python -m traffic_ai.video_pipeline` on `data/sample/cars.mp4` (1280×720, 368 frames):

* 63 vehicles counted across 3 lanes (cars, bus, truck), with per-lane breakdown in `results/video_summary.json`
* ~7.5 FPS with `yolov8n` on a 2-core CPU, no GPU. A GPU or `yolov8s/m` gives real-time speed and higher accuracy.

## Quick start

```bash
git clone https://github.com/Drakeion2532/Traffic-Management-using-Deep-Learning.git
cd Traffic-Management-using-Deep-Learning
pip install -r requirements.txt

# 1) count vehicles + recommend green time on the sample video
python -m traffic_ai.video_pipeline --video data/sample/cars.mp4 --config configs/cars_sample.yaml
#    -> results/annotated.mp4, results/density_log.csv, results/video_summary.json

# 2) live webcam / RTSP CCTV stream
python -m traffic_ai.video_pipeline --video 0 --show
python -m traffic_ai.video_pipeline --video rtsp://user:pass@camera-ip/stream --config configs/my_junction.yaml

# 3) static-vs-adaptive benchmark
python -m traffic_ai.simulation

# 4) tests
pytest -q
```

YOLO weights (`yolov8n.pt` etc.) download automatically on first run, so they are no longer committed to the repo.

### Using your own camera

Copy `configs/cars_sample.yaml` and set:
* `roi`: polygon of the road you care about (pixels)
* `count_line`: y position and x-extent of the virtual stop line
* `lanes`: x-ranges of each lane along that line

## Dataset

* `data/sample/cars.mp4`: sample traffic footage used for the demo.
* Detection uses COCO-pretrained YOLOv8, whose vehicle classes cover Indian traffic reasonably well. To add auto-rickshaws or emergency vehicles, fine-tune on a labelled set (e.g. [IDD – India Driving Dataset](https://idd.insaan.iiit.ac.in/)) with `yolo detect train data=your.yaml model=yolov8n.pt`.

## Roadmap (from the paper's future scope)

- [ ] Emergency-vehicle priority (fine-tuned ambulance class → immediate green)
- [ ] Red-light violation detection using the counting line + signal state
- [ ] Accident / breakdown detection (stationary track inside ROI for > N seconds)
- [ ] Green-wave synchronisation across consecutive junctions

## Project structure

```
traffic_ai/
  signal_timer.py     adaptive green-time logic
  simulation.py       static vs adaptive junction simulator
  video_pipeline.py   YOLOv8 + ByteTrack counting pipeline
configs/              junction geometry (ROI, count line, lanes)
data/sample/          sample video
docs/                 demo gif + result plots
results/              generated outputs
tests/                pytest suite
```
