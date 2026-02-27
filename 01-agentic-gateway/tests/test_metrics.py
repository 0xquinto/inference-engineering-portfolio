"""Tests for the metrics collector."""

from src.dashboard.metrics import MetricsCollector, RequestMetric


def make_metric(model_key="small", input_tokens=100, output_tokens=50, total_time_ms=200.0, cost_usd=0.000015):
    return RequestMetric(
        timestamp=1000.0,
        model_key=model_key,
        complexity_score=0.3,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ttft_ms=50.0,
        total_time_ms=total_time_ms,
        had_tool_calls=False,
        cost_usd=cost_usd,
    )


def test_empty_summary():
    mc = MetricsCollector()
    summary = mc.summary()
    assert summary["total_requests"] == 0


def test_single_request_summary():
    mc = MetricsCollector()
    mc.record(make_metric())
    summary = mc.summary()
    assert summary["total_requests"] == 1
    assert summary["routing_split"]["small"] == 1
    assert summary["routing_split"]["large"] == 0


def test_cost_savings_calculation():
    mc = MetricsCollector()
    # Record a cheap small-model request
    mc.record(make_metric(model_key="small", input_tokens=1000, output_tokens=500, cost_usd=0.00015))
    summary = mc.summary()
    # if_all_large should be more expensive: (1000*0.80 + 500*0.80) / 1_000_000 = 0.0012
    assert summary["cost"]["if_all_large_usd"] > summary["cost"]["total_usd"]
    assert summary["cost"]["savings_usd"] > 0
    assert summary["cost"]["savings_pct"] > 0


def test_latency_percentiles():
    mc = MetricsCollector()
    for ms in [100.0, 200.0, 300.0, 400.0, 500.0]:
        mc.record(make_metric(total_time_ms=ms))
    summary = mc.summary()
    assert summary["latency_ms"]["p50"] > 0
    assert summary["latency_ms"]["p95"] >= summary["latency_ms"]["p50"]


def test_routing_split_percentages():
    mc = MetricsCollector()
    mc.record(make_metric(model_key="small"))
    mc.record(make_metric(model_key="small"))
    mc.record(make_metric(model_key="large"))
    summary = mc.summary()
    assert summary["routing_split"]["small"] == 2
    assert summary["routing_split"]["large"] == 1
    assert abs(summary["routing_split"]["small_pct"] - 66.7) < 1.0


def test_tool_call_tracking():
    mc = MetricsCollector()
    m = make_metric()
    m.had_tool_calls = True
    mc.record(m)
    mc.record(make_metric())  # no tool calls
    summary = mc.summary()
    assert summary["tool_calls"]["total"] == 1
    assert summary["tool_calls"]["pct"] == 50.0
