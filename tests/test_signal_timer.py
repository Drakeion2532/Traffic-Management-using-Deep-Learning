from traffic_ai.signal_timer import AdaptiveSignalTimer, SignalTimerConfig
from traffic_ai.simulation import simulate, run_benchmark, APPROACHES


def test_empty_approach_gets_min_green():
    t = AdaptiveSignalTimer()
    assert t.green_time({}) == t.cfg.min_green


def test_green_grows_with_traffic_and_is_capped():
    t = AdaptiveSignalTimer(SignalTimerConfig(lanes=2))
    small = t.green_time({"car": 5})
    big = t.green_time({"car": 20, "bus": 3})
    huge = t.green_time({"truck": 500})
    assert small < big <= t.cfg.max_green
    assert huge == t.cfg.max_green


def test_heavy_vehicles_need_more_time_than_bikes():
    t = AdaptiveSignalTimer()
    assert t.green_time({"bus": 12}) > t.green_time({"motorbike": 12})


def test_more_lanes_clear_faster():
    t = AdaptiveSignalTimer()
    assert t.green_time({"car": 30}, lanes=4) < t.green_time({"car": 30}, lanes=1)


def test_density_ranking():
    t = AdaptiveSignalTimer()
    order = t.density_ranking({"n": {"car": 2}, "e": {"bus": 3}, "s": {"motorbike": 1}})
    assert order == ["e", "n", "s"]


def test_simulation_is_deterministic_and_adaptive_not_worse():
    rates = dict(zip(APPROACHES, [0.5, 0.1, 0.1, 0.1]))
    a1 = simulate("adaptive", rates, 300, seed=3)
    a2 = simulate("adaptive", rates, 300, seed=3)
    s = simulate("static", rates, 300, seed=3)
    assert a1 == a2
    assert a1.throughput >= s.throughput
    assert a1.idle_green_s <= s.idle_green_s


def test_benchmark_shape():
    res = run_benchmark(n_sims=2, duration=120)
    assert len(res["runs"]) == 2
    assert "throughput_gain_pct" in res["summary"]
