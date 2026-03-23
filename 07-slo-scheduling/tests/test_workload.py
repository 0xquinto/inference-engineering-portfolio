import pytest

from src.config import WorkloadClass
from src.workload import WorkloadGenerator, WorkloadRequest


class TestWorkloadRequest:
    def test_fields(self):
        r = WorkloadRequest(
            prompt="hello", request_class="short", slo_seconds=2.0,
            priority=0, deadline=1000.0,
        )
        assert r.request_class == "short"
        assert r.slo_seconds == 2.0
        assert r.priority == 0

    def test_is_short(self):
        r = WorkloadRequest("hi", "short", 2.0, 0, 100.0)
        assert r.request_class == "short"


class TestWorkloadGenerator:
    @pytest.fixture
    def classes(self):
        return [
            WorkloadClass("short", 0.4, 50, 2.0, "short prompt"),
            WorkloadClass("medium", 0.4, 200, 8.0, "medium prompt"),
            WorkloadClass("long", 0.2, 500, 20.0, "long prompt"),
        ]

    def test_generate_count(self, classes):
        gen = WorkloadGenerator(classes)
        requests = gen.generate(100)
        assert len(requests) == 100

    def test_class_distribution(self, classes):
        gen = WorkloadGenerator(classes, seed=42)
        requests = gen.generate(1000)
        short_count = sum(1 for r in requests if r.request_class == "short")
        medium_count = sum(1 for r in requests if r.request_class == "medium")
        long_count = sum(1 for r in requests if r.request_class == "long")
        assert 350 < short_count < 450
        assert 350 < medium_count < 450
        assert 150 < long_count < 250

    def test_deadlines_set(self, classes):
        gen = WorkloadGenerator(classes)
        requests = gen.generate(10)
        for r in requests:
            assert r.deadline > 0
            assert r.slo_seconds > 0

    def test_priority_by_class(self, classes):
        gen = WorkloadGenerator(classes)
        requests = gen.generate(100)
        for r in requests:
            if r.request_class == "short":
                assert r.priority == 0
            elif r.request_class == "medium":
                assert r.priority == 1
            elif r.request_class == "long":
                assert r.priority == 2

    def test_deterministic_with_seed(self, classes):
        gen1 = WorkloadGenerator(classes, seed=42)
        gen2 = WorkloadGenerator(classes, seed=42)
        r1 = gen1.generate(50)
        r2 = gen2.generate(50)
        assert [r.request_class for r in r1] == [r.request_class for r in r2]

    def test_prompts_from_config(self, classes):
        gen = WorkloadGenerator(classes)
        requests = gen.generate(10)
        for r in requests:
            assert len(r.prompt) > 0
