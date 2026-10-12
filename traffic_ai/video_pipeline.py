"""YOLOv8 + ByteTrack video pipeline.

For each frame:
  1. detect vehicles (COCO car / motorcycle / bus / truck / bicycle),
  2. track them with ByteTrack so each vehicle keeps an ID,
  3. count unique vehicles crossing a virtual line, split by lane,
  4. measure live density (vehicles inside the region of interest, in PCU),
  5. feed the live per-class count to the AdaptiveSignalTimer and overlay
     the recommended green time.

Outputs: annotated MP4, per-second CSV log, JSON summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from .signal_timer import AdaptiveSignalTimer, SignalTimerConfig

# COCO class id -> our vehicle class
COCO_TO_VEHICLE = {1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
COLORS = {"car": (255, 128, 0), "motorbike": (0, 200, 255), "bus": (0, 0, 255),
          "truck": (180, 0, 180), "bicycle": (0, 255, 0)}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _crossed(prev_y, cur_y, line_y):
    return (prev_y - line_y) * (cur_y - line_y) <= 0 and prev_y != cur_y


def run(video, config, out_dir="results", weights="yolov8n.pt", max_frames=None, show=False):
    from ultralytics import YOLO  # imported lazily so the timer/sim work without torch

    cfg = load_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    timer = AdaptiveSignalTimer(SignalTimerConfig(lanes=len(cfg["lanes"])))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    roi = np.array(cfg["roi"], dtype=np.int32)
    roi_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(roi_mask, [roi], 255)
    line_y = cfg["count_line"]["y"]
    lx1, lx2 = cfg["count_line"]["x1"], cfg["count_line"]["x2"]
    lanes = cfg["lanes"]  # list of {name, x_min, x_max}

    last_y = {}
    counted = set()
    lane_counts = defaultdict(lambda: defaultdict(int))
    log_rows, frame_i, t0 = [], 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and frame_i >= max_frames):
            break
        res = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False,
                          classes=list(COCO_TO_VEHICLE), conf=cfg.get("conf", 0.3), imgsz=cfg.get("imgsz", 640))[0]
        live = defaultdict(int)
        if res.boxes is not None and res.boxes.id is not None:
            for box, tid, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().cpu().tolist(),
                                     res.boxes.cls.int().cpu().tolist()):
                vcls = COCO_TO_VEHICLE.get(cls)
                if vcls is None:
                    continue
                x1, y1, x2, y2 = box.astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if not (0 <= cy < h and 0 <= cx < w) or roi_mask[cy, cx] == 0:
                    continue
                live[vcls] += 1
                if tid in last_y and tid not in counted and lx1 <= cx <= lx2 and _crossed(last_y[tid], cy, line_y):
                    counted.add(tid)
                    lane = next((l["name"] for l in lanes if l["x_min"] <= cx < l["x_max"]), "other")
                    lane_counts[lane][vcls] += 1
                last_y[tid] = cy
                c = COLORS[vcls]
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                cv2.putText(frame, f"{vcls} #{tid}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        green = timer.green_time(live)
        pcu = timer.density_pcu(live)
        total = sum(sum(v.values()) for v in lane_counts.values())
        cv2.polylines(frame, [roi], True, (0, 255, 255), 1)
        cv2.line(frame, (lx1, line_y), (lx2, line_y), (0, 0, 255), 3)
        for l in lanes:
            cv2.line(frame, (l["x_min"], line_y - 12), (l["x_min"], line_y + 12), (255, 255, 255), 2)
        panel = [f"Counted: {total}", f"In ROI: {sum(live.values())}  ({pcu:.1f} PCU)",
                 f"Recommended green: {green:.0f}s"] + \
                [f"{ln}: {sum(c.values())}" for ln, c in sorted(lane_counts.items())]
        cv2.rectangle(frame, (10, 10), (330, 30 + 24 * len(panel)), (0, 0, 0), -1)
        for i, txt in enumerate(panel):
            cv2.putText(frame, txt, (20, 35 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(frame)
        if show:
            cv2.imshow("traffic", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if frame_i % int(fps) == 0:
            log_rows.append({"t_s": round(frame_i / fps, 2), "in_roi": sum(live.values()),
                             "pcu": round(pcu, 2), "green_s": green, "counted_total": total})
        frame_i += 1

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    elapsed = time.time() - t0
    with open(out / "density_log.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(log_rows[0]) if log_rows else ["t_s"])
        wr.writeheader()
        wr.writerows(log_rows)
    summary = {"video": str(video), "frames": frame_i, "processing_fps": round(frame_i / elapsed, 2),
               "total_counted": sum(sum(v.values()) for v in lane_counts.values()),
               "per_lane": {k: dict(v) for k, v in sorted(lane_counts.items())},
               "peak_in_roi": max((r["in_roi"] for r in log_rows), default=0)}
    with open(out / "video_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser(description="YOLOv8 vehicle counting + adaptive signal recommendation")
    p.add_argument("--video", default="data/sample/cars.mp4", help="video file, RTSP URL or webcam index")
    p.add_argument("--config", default="configs/cars_sample.yaml")
    p.add_argument("--weights", default="yolov8n.pt", help="yolov8n/s/m/l .pt (auto-downloaded)")
    p.add_argument("--out", default="results")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    src = int(a.video) if a.video.isdigit() else a.video
    print(json.dumps(run(src, a.config, a.out, a.weights, a.max_frames, a.show), indent=2))


if __name__ == "__main__":
    main()
