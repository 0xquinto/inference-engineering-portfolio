"""Tests for the routing strategy."""

from src.router.classifier import ComplexityClassifier
from src.router.strategy import RoutingStrategy


def make_strategy(available=None, threshold=0.6):
    classifier = ComplexityClassifier(threshold=threshold)
    return RoutingStrategy(classifier, available or ["small", "large"])


def test_explicit_model_override():
    strategy = make_strategy()
    key, reason = strategy.select_model(
        messages=[{"role": "user", "content": "Hi"}],
        requested_model="large",
    )
    assert key == "large"
    assert "Explicitly requested" in reason


def test_explicit_override_maps_70b_to_large():
    strategy = make_strategy()
    key, _ = strategy.select_model(
        messages=[{"role": "user", "content": "Hi"}],
        requested_model="my-70b-model",
    )
    assert key == "large"


def test_explicit_override_maps_8b_to_small():
    strategy = make_strategy()
    key, _ = strategy.select_model(
        messages=[{"role": "user", "content": "Hi"}],
        requested_model="some-8b-variant",
    )
    assert key == "small"


def test_simple_message_routes_to_small():
    strategy = make_strategy()
    key, reason = strategy.select_model(
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert key == "small"
    assert "Simple" in reason


def test_complex_message_routes_to_large():
    strategy = make_strategy()
    complex_msg = (
        "Analyze and compare the performance characteristics of quicksort versus mergesort "
        "across different input distributions including sorted, reverse sorted, and random data. "
        "Write code to implement both algorithms and debug any issues you find. "
        "Evaluate which is more suitable for large-scale production workloads."
    )
    key, reason = strategy.select_model(
        messages=[{"role": "user", "content": complex_msg}],
        tools=[{"type": "function", "function": {"name": "run_code"}}],
        response_format={"type": "json_object"},
    )
    assert key == "large"
    assert "Complex" in reason


def test_fallback_when_large_unavailable():
    strategy = make_strategy(available=["small"])
    complex_msg = (
        "Analyze and compare the performance characteristics of quicksort versus mergesort "
        "across different input distributions including sorted, reverse sorted, and random data. "
        "Write code to implement both algorithms and debug any issues you find. "
        "Evaluate which is more suitable for large-scale production workloads."
    )
    key, reason = strategy.select_model(
        messages=[{"role": "user", "content": complex_msg}],
        tools=[{"type": "function", "function": {"name": "run_code"}}],
        response_format={"type": "json_object"},
    )
    assert key == "small"
    assert "Fallback" in reason


def test_fallback_when_small_unavailable():
    strategy = make_strategy(available=["large"])
    key, reason = strategy.select_model(
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert key == "large"
    assert "Fallback" in reason
