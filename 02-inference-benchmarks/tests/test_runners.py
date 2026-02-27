"""Tests for benchmark runners (no GPU required — tests runner logic only)."""

import pytest
from src.runners.base import BenchmarkRunner, BenchmarkResult, RequestConfig


def test_benchmark_result_tokens_per_sec():
    r = BenchmarkResult(
        engine="vllm",
        prompt_category="short",
        concurrency=1,
        ttft_ms=50.0,
        total_time_ms=1000.0,
        tokens_generated=100,
        tokens_per_sec=100.0,
        gpu_memory_mb=45000.0,
    )
    assert r.tokens_per_sec == 100.0
    assert r.engine == "vllm"


def test_request_config_defaults():
    cfg = RequestConfig()
    assert cfg.max_tokens == 256
    assert cfg.temperature == 0.0


def test_benchmark_runner_is_abstract():
    with pytest.raises(TypeError):
        BenchmarkRunner()
