"""Tests that all runners can be imported and instantiated."""

from src.runners import VllmRunner
from src.runners.sglang_runner import SglangRunner
from src.runners.trtllm_runner import TrtllmRunner


def test_vllm_runner_init():
    r = VllmRunner(port=8001)
    assert r.engine_name == "vllm"
    assert r.port == 8001


def test_sglang_runner_init():
    r = SglangRunner(port=8002)
    assert r.engine_name == "sglang"
    assert r.port == 8002


def test_trtllm_runner_init():
    r = TrtllmRunner(port=8003)
    assert r.engine_name == "tensorrt-llm"
    assert r.port == 8003
