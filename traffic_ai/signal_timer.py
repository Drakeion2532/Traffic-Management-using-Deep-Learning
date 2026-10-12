"""Adaptive green-signal timing.

Implements the signal-switching logic described in
"AI-Powered Traffic Optimization with YOLOv8 Object Detection"
(Phatangare, Sakpal, Khan, Punde, Kasurde):

* green time is computed from the per-class vehicle count waiting on the
  approach that is about to turn green,
* each vehicle class has an average time to clear the stop line,
* lanes discharge in parallel, so the load is divided across lanes,
* a start-up lag is added for the first vehicle in the queue,
* min / max limits prevent lane starvation,
* the signal ORDER stays fixed (N -> E -> S -> W); only DURATIONS adapt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping

# Average seconds for one vehicle of each class to cross the stop line
# from a standing queue (typical urban values; tune per junction).
DEFAULT_CLEAR_TIME: Dict[str, float] = {
    "car": 2.0,
    "motorbike": 1.0,
    "bicycle": 1.5,
    "auto_rickshaw": 2.25,
    "bus": 2.5,
    "truck": 2.5,
}

# Passenger Car Units – used for density reporting.
PCU: Dict[str, float] = {
    "car": 1.0,
    "motorbike": 0.5,
    "bicycle": 0.5,
    "auto_rickshaw": 0.75,
    "bus": 3.0,
    "truck": 3.0,
}


@dataclass
class SignalTimerConfig:
    min_green: float = 10.0
    max_green: float = 60.0
    default_green: float = 20.0
    yellow: float = 3.0
    startup_lag: float = 2.0
    lanes: int = 2
    clear_time: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CLEAR_TIME))


class AdaptiveSignalTimer:
    """Computes green durations from detected vehicle counts."""

    def __init__(self, config: SignalTimerConfig | None = None):
        self.cfg = config or SignalTimerConfig()

    def green_time(self, counts: Mapping[str, int], lanes: int | None = None) -> float:
        """Green time (seconds) for one approach given its waiting vehicles."""
        lanes = lanes or self.cfg.lanes
        total = sum(counts.values())
        if total == 0:
            return self.cfg.min_green
        load = sum(n * self.cfg.clear_time.get(cls, 2.0) for cls, n in counts.items())
        # lanes discharge in parallel; +1 mirrors the conservative divisor used
        # in the paper's timing rule so the estimate never under-serves a queue.
        gst = math.ceil(load / (lanes + 1)) + self.cfg.startup_lag
        return float(min(self.cfg.max_green, max(self.cfg.min_green, gst)))

    @staticmethod
    def density_pcu(counts: Mapping[str, int]) -> float:
        return sum(n * PCU.get(cls, 1.0) for cls, n in counts.items())

    def plan_cycle(self, approach_counts: Mapping[str, Mapping[str, int]]) -> Dict[str, float]:
        """Green time for every approach (fixed order preserved by dict order)."""
        return {name: self.green_time(c) for name, c in approach_counts.items()}

    def density_ranking(self, approach_counts: Mapping[str, Mapping[str, int]]):
        """Approaches sorted by PCU density (high -> low), for dashboards/priority."""
        return sorted(approach_counts, key=lambda a: self.density_pcu(approach_counts[a]), reverse=True)
