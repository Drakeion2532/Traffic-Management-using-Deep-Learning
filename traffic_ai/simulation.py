"""Headless 4-way intersection simulator: static timer vs adaptive timer.

Mirrors the paper's evaluation protocol: 12 simulations of 5 minutes each,
varied traffic patterns in all four directions, metric = vehicles crossing
the junction (throughput), plus idle green time and average wait.

Model (1-second time step):
* arrivals per approach ~ Poisson(rate), class mix sampled per vehicle
* queue discharges only on green; `lanes` vehicles can move in parallel and
  each vehicle needs its class clear-time; first vehicle pays a start-up lag
* adaptive controller reads the queue the way the camera would: each vehicle
  is detected with probability `detection_recall` (YOLO is not perfect)
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np

from .signal_timer import AdaptiveSignalTimer, SignalTimerConfig, PCU

APPROACHES = ["north", "east", "south", "west"]
CLASS_MIX = {"car": 0.45, "motorbike": 0.30, "auto_rickshaw": 0.10, "bus": 0.08, "truck": 0.07}


@dataclass
class SimResult:
    controller: str
    throughput: int
    avg_wait_s: float           # vehicles that crossed (as in the paper)
    idle_green_s: float
    remaining_queue: int
    avg_delay_all_s: float = 0.0   # every vehicle, incl. those still queued at the end (fairer)
    max_red_s: float = 0.0         # longest any approach with waiting vehicles sat on red


def _sample_class(rng):
    return rng.choice(list(CLASS_MIX), p=list(CLASS_MIX.values()))


def simulate(controller: str, rates: Dict[str, float], duration: int = 300, seed: int = 0,
             static_green: float = 30.0, detection_recall: float = 0.9,
             cfg: SignalTimerConfig | None = None, max_red: float = 90.0) -> SimResult:
    """controller: "static" (fixed 30 s, fixed order), "adaptive" (paper: camera-based durations, fixed
    order) or "demand" (camera-based durations AND the next green goes to the busiest approach, skipping
    empty ones, but no approach with waiting vehicles is kept on red longer than `max_red` seconds)."""
    cfg = cfg or SignalTimerConfig()
    timer = AdaptiveSignalTimer(cfg)
    rng = np.random.default_rng(seed)
    queues = {a: deque() for a in APPROACHES}  # items: (class, arrival_t)
    # identical arrival stream for both controllers (same seed)
    arrivals = {a: rng.poisson(rates[a], size=duration) for a in APPROACHES}
    classes = {a: [[_sample_class(rng) for _ in range(n)] for n in arrivals[a]] for a in APPROACHES}
    det_rng = np.random.default_rng(seed + 1)

    phase_idx, phase_left, in_yellow = 0, 0.0, False
    green_elapsed, capacity = 0.0, 0.0
    crossed, waits, idle = 0, [], 0.0
    red_since = {a: 0 for a in APPROACHES}     # when each approach last lost its green
    worst_red = 0.0

    def detected_pcu(a):
        return sum(PCU.get(cls, 1.0) for cls, _ in queues[a] if det_rng.random() < detection_recall)

    def next_phase(cur, t):
        if controller != "demand":
            return (cur + 1) % len(APPROACHES)
        others = [i for i in range(len(APPROACHES)) if i != cur]
        # starvation guard: anyone with waiting vehicles who has been red too long goes first
        overdue = [i for i in others if queues[APPROACHES[i]] and t - red_since[APPROACHES[i]] >= max_red]
        if overdue:
            return max(overdue, key=lambda i: t - red_since[APPROACHES[i]])
        load = {i: detected_pcu(APPROACHES[i]) for i in others}
        best = max(load, key=load.get)
        return best if load[best] > 0 else (cur + 1) % len(APPROACHES)

    def start_green(idx):
        a = APPROACHES[idx]
        if controller == "static":
            return static_green
        seen: Dict[str, int] = {}
        for cls, _ in queues[a]:
            if det_rng.random() < detection_recall:
                seen[cls] = seen.get(cls, 0) + 1
        return timer.green_time(seen)

    phase_left = start_green(0)
    for t in range(duration):
        for a in APPROACHES:
            for cls in classes[a][t]:
                queues[a].append((cls, t))
        active = APPROACHES[phase_idx]
        if not in_yellow:
            if green_elapsed >= cfg.startup_lag:
                capacity += cfg.lanes
                q = queues[active]
                if not q:
                    idle += 1
                    capacity = 0.0
                while q and capacity >= cfg.clear_time.get(q[0][0], 2.0):
                    cls, at = q.popleft()
                    capacity -= cfg.clear_time.get(cls, 2.0)
                    crossed += 1
                    waits.append(t - at)
            green_elapsed += 1
            # demand mode: end this green early (after the minimum green) if another approach with
            # waiting vehicles is about to exceed the red-time cap
            if controller == "demand" and green_elapsed >= cfg.min_green and phase_left > 1:
                if any(queues[o] and t + cfg.yellow - red_since[o] >= max_red for o in APPROACHES if o != active):
                    phase_left = 1
        phase_left -= 1
        if phase_left <= 0:
            if not in_yellow:
                in_yellow, phase_left = True, cfg.yellow
            else:
                in_yellow = False
                red_since[APPROACHES[phase_idx]] = t
                phase_idx = next_phase(phase_idx, t)
                if queues[APPROACHES[phase_idx]]:
                    worst_red = max(worst_red, t - red_since[APPROACHES[phase_idx]])
                green_elapsed, capacity = 0.0, 0.0
                phase_left = start_green(phase_idx)
    still_waiting = [duration - at for q in queues.values() for _, at in q]
    for a in APPROACHES:   # approaches still red at the end count toward the longest red
        if a != APPROACHES[phase_idx] and queues[a]:
            worst_red = max(worst_red, duration - red_since[a])
    return SimResult(controller, crossed, float(np.mean(waits)) if waits else 0.0, idle,
                     sum(len(q) for q in queues.values()),
                     float(np.mean(waits + still_waiting)) if waits or still_waiting else 0.0, float(worst_red))


def run_benchmark(n_sims: int = 12, duration: int = 300, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for i in range(n_sims):
        # varied, unbalanced demand: one or two busy approaches per scenario
        base = rng.uniform(0.05, 0.25, size=4)
        busy = rng.choice(4, size=rng.integers(1, 3), replace=False)
        base[busy] *= rng.uniform(2.0, 3.5)
        rates = dict(zip(APPROACHES, base.round(3)))
        s = simulate("static", rates, duration, seed=i)
        a = simulate("adaptive", rates, duration, seed=i)
        d = simulate("demand", rates, duration, seed=i)
        rows.append({"sim": i + 1, "rates_veh_per_s": rates,
                     "static": asdict(s), "adaptive": asdict(a), "demand": asdict(d),
                     "throughput_gain_pct": round(100 * (a.throughput - s.throughput) / max(1, s.throughput), 2)})
    tot_s = sum(r["static"]["throughput"] for r in rows)
    tot_a = sum(r["adaptive"]["throughput"] for r in rows)
    summary = {
        "simulations": n_sims, "duration_s": duration,
        "static_total_throughput": tot_s, "adaptive_total_throughput": tot_a,
        "throughput_gain_pct": round(100 * (tot_a - tot_s) / tot_s, 2),
        "static_avg_wait_s": round(float(np.mean([r["static"]["avg_wait_s"] for r in rows])), 2),
        "adaptive_avg_wait_s": round(float(np.mean([r["adaptive"]["avg_wait_s"] for r in rows])), 2),
        "static_idle_green_s": round(float(np.mean([r["static"]["idle_green_s"] for r in rows])), 1),
        "adaptive_idle_green_s": round(float(np.mean([r["adaptive"]["idle_green_s"] for r in rows])), 1),
    }
    # fairer comparison, incl. vehicles still queued at the end, plus the demand-ordered controller
    for c in ("static", "adaptive", "demand"):
        summary[f"{c}_throughput"] = sum(r[c]["throughput"] for r in rows)
        summary[f"{c}_avg_delay_all_vehicles_s"] = round(float(np.mean([r[c]["avg_delay_all_s"] for r in rows])), 1)
        summary[f"{c}_left_queued_at_end"] = round(float(np.mean([r[c]["remaining_queue"] for r in rows])), 1)
        summary[f"{c}_worst_red_s"] = round(float(max(r[c]["max_red_s"] for r in rows)), 0)
    return {"summary": summary, "runs": rows}


def main():
    p = argparse.ArgumentParser(description="Static vs adaptive signal benchmark")
    p.add_argument("--sims", type=int, default=12)
    p.add_argument("--duration", type=int, default=300)
    p.add_argument("--out", default="results/simulation_results.json")
    args = p.parse_args()
    res = run_benchmark(args.sims, args.duration)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res["summary"], indent=2))
    print(f"\n{'sim':>3} {'static':>7} {'adaptive':>9} {'demand':>7} {'gain%':>7}")
    for r in res["runs"]:
        print(f"{r['sim']:>3} {r['static']['throughput']:>7} {r['adaptive']['throughput']:>9} "
              f"{r['demand']['throughput']:>7} {r['throughput_gain_pct']:>7}")


if __name__ == "__main__":
    main()
