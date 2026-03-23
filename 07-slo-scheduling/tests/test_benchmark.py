from src.benchmark import SchedulingBenchmarker


class TestSchedulingBenchmarker:
    def test_init_default(self):
        b = SchedulingBenchmarker(port=8010, model_name="test")
        assert b.base_url == "http://localhost:8010"
        assert b.policy == "fcfs"
        assert b.scheduler is None

    def test_init_slo_aware_creates_scheduler(self):
        b = SchedulingBenchmarker(
            port=8010, model_name="test", policy="slo_aware",
            scheduler_config={"max_queue_depth": 20, "max_concurrent": 5},
        )
        assert b.scheduler is not None
        assert b.scheduler.max_queue_depth == 20
        assert b.scheduler.max_concurrent == 5

    def test_init_priority_no_scheduler(self):
        b = SchedulingBenchmarker(port=8010, model_name="test", policy="priority")
        assert b.scheduler is None
