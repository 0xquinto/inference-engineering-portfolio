from pathlib import Path

from src.config import WorkloadClass, SchedulerConfig, SchedulingConfig, load_config


class TestWorkloadClass:
    def test_from_dict(self):
        data = {"share": 0.4, "output_tokens": 50, "slo_seconds": 2.0,
                "prompt": "Hello"}
        wc = WorkloadClass.from_dict("short", data)
        assert wc.name == "short"
        assert wc.share == 0.4
        assert wc.output_tokens == 50
        assert wc.slo_seconds == 2.0

    def test_from_dict_all_fields(self):
        data = {"share": 0.2, "output_tokens": 500, "slo_seconds": 20.0,
                "prompt": "Write code"}
        wc = WorkloadClass.from_dict("long", data)
        assert wc.name == "long"
        assert wc.prompt == "Write code"


class TestSchedulerConfig:
    def test_defaults(self):
        sc = SchedulerConfig()
        assert sc.max_queue_depth == 50
        assert sc.max_concurrent == 10

    def test_custom(self):
        sc = SchedulerConfig(max_queue_depth=20, max_concurrent=3)
        assert sc.max_queue_depth == 20


class TestLoadConfig:
    def test_loads_yaml(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "scheduling.yaml")
        assert isinstance(cfg, SchedulingConfig)
        assert cfg.model_name == "Qwen/Qwen3.5-9B"

    def test_workload_classes(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "scheduling.yaml")
        assert len(cfg.workload_classes) == 3
        names = [wc.name for wc in cfg.workload_classes]
        assert "short" in names
        assert "medium" in names
        assert "long" in names

    def test_shares_sum_to_one(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "scheduling.yaml")
        total = sum(wc.share for wc in cfg.workload_classes)
        assert abs(total - 1.0) < 0.01

    def test_policies(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "scheduling.yaml")
        assert "fcfs" in cfg.policies
        assert "slo_aware" in cfg.policies

    def test_qps_levels(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "scheduling.yaml")
        assert cfg.qps_levels == [1, 5, 10, 20]

    def test_scheduler_config(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "scheduling.yaml")
        assert cfg.scheduler.max_queue_depth == 50
        assert cfg.scheduler.max_concurrent == 10
