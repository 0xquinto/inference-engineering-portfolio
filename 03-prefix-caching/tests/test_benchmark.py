import pytest

from src.benchmark import CacheBenchmarker


class TestCacheBenchmarker:
    def test_init(self):
        b = CacheBenchmarker(port=8010, max_tokens=128)
        assert b.base_url == "http://localhost:8010"

    def test_build_payload_single_turn(self):
        b = CacheBenchmarker(port=8010)
        system = {"role": "system", "content": "You are helpful."}
        user = {"role": "user", "content": "Hello"}
        payload = b._build_payload([system, user])
        assert payload["messages"] == [system, user]
        assert payload["stream"] is True

    def test_build_payload_multi_turn(self):
        b = CacheBenchmarker(port=8010)
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
        ]
        payload = b._build_payload(messages)
        assert len(payload["messages"]) == 3
