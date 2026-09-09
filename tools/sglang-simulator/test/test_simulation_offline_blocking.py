import json

import pytest
from test_simulation_sglang_serving import (
    SIM_CONFIGS,
    SGLangServingRunner,
    assert_decode_metrics,
)

REQUEST_RATE = 1
SEED = 123
# OFFLINE replays the seeded arrival schedule on a virtual clock; BLOCKING
# realizes it with real sleeps and timestamps each arrival when the server
# ingests it. The first request also pays a cold ingest cost the later ones do
# not, and every arrival is reported relative to it, so BLOCKING lands
# consistently early rather than symmetrically off. Measured over 12 runs on a
# contended 4-core host: 7.1 ms - 22.5 ms early, always early.
ARRIVAL_TOLERANCE_S = 0.05
# `duration` spans that timeline (last event minus first arrival) and the three
# throughputs are constants divided by it, so all four carry the same drift. The
# span is ~1.5 s here, so ARRIVAL_TOLERANCE_S alone is worth ~3%. Measured over
# the same 12 runs: 0.28% - 1.48%.
DURATION_TOLERANCE = 0.05
# The latency means cover 3 requests of 4 output tokens each, so the same
# millisecond drift lands on a far smaller denominator. Measured over the same
# 12 runs: mean_ttft_ms 4.3% - 12.1%, the other three up to 9.5%.
LATENCY_TOLERANCE = 0.25
RELATIVE_TOLERANCES = {
    "duration": DURATION_TOLERANCE,
    "request_throughput": DURATION_TOLERANCE,
    "input_throughput": DURATION_TOLERANCE,
    "output_throughput": DURATION_TOLERANCE,
    "mean_e2e_latency_ms": LATENCY_TOLERANCE,
    "mean_ttft_ms": LATENCY_TOLERANCE,
    "mean_tpot_ms": LATENCY_TOLERANCE,
    "mean_itl_ms": LATENCY_TOLERANCE,
}


def _relative_error(actual, expected):
    return abs(actual - expected) / abs(expected)


def _run_mode(mode, tmp_path):
    case_dir = tmp_path / mode
    case_dir.mkdir()
    runner = SGLangServingRunner(SIM_CONFIGS["aic_sol"], case_dir, mode=mode)
    try:
        metrics = runner.benchmark(
            case_dir / "benchmark.json", request_rate=REQUEST_RATE, seed=SEED
        )
    finally:
        runner.shutdown()

    requests = [
        json.loads(line)
        for line in (runner.output_dir / "request.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    requests.sort(key=lambda request: request["created_time"])
    return metrics, requests


def test_request_rate_offline_matches_blocking(tmp_path):
    offline_metrics, offline_requests = _run_mode("offline", tmp_path)
    blocking_metrics, blocking_requests = _run_mode("blocking", tmp_path)

    for metrics in (offline_metrics, blocking_metrics):
        assert_decode_metrics(metrics)

    assert len(offline_requests) == len(blocking_requests) == 3

    offline_arrivals = [request["created_time"] for request in offline_requests]
    blocking_arrivals = [request["created_time"] for request in blocking_requests]
    assert offline_arrivals[1] > 0.5
    assert blocking_arrivals[1] > 0.5
    assert offline_arrivals == pytest.approx(blocking_arrivals, abs=ARRIVAL_TOLERANCE_S)
    assert (
        offline_metrics["max_concurrent_requests"]
        == blocking_metrics["max_concurrent_requests"]
        == 1
    )

    for key in ("completed", "total_input", "total_output"):
        assert offline_metrics[key] == blocking_metrics[key]

    for key, tolerance in RELATIVE_TOLERANCES.items():
        error = _relative_error(offline_metrics[key], blocking_metrics[key])
        assert error <= tolerance, (
            key,
            offline_metrics[key],
            blocking_metrics[key],
            error,
            tolerance,
        )
