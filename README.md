# AI-Powered Traffic Optimization with YOLOv8

Real-time vehicle detection, tracking, lane-wise counting and **adaptive traffic-signal timing** from ordinary CCTV footage.

This is the reference implementation of the paper
**"AI-Powered Traffic Optimization with YOLOv8 Object Detection"** by
S. Phatangare, R. Sakpal, S. Khan, **S. Punde**, S. Kasurde (Vishwakarma Institute of Technology, Pune). *(Add the IEEE Xplore DOI link here.)*

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
| Avg. wait per vehicle | 64.8 s | 62.6 s |

![simulation](docs/simulation_results.png)

The roughly 14% throughput gain matches the ≈15% improvement over static control reported in the paper's simulations.

### 2. Real video

`python -m traffic_ai.video_pipeline` on `data/sample/cars.mp4` (1280×720, 368 frames):

* 63 vehicles counted across 4 lanes (cars, bus, truck), with per-lane breakdown in `results/video_summary.json`
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
