from pathlib import Path

from src.config import ModelTier, CostConfig, load_config


class TestModelTier:
    def test_from_dict_small(self):
        data = {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "params": "0.5B",
            "description": "Fast, cheap, handles simple queries",
            "port": 8010,
            "gpu_cost_per_hour": 0.75,
            "vram_mb": 1200,
        }
        tier = ModelTier.from_dict("small", data)
        assert tier.name == "small"
        assert tier.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert tier.port == 8010
        assert tier.gpu_cost_per_hour == 0.75
        assert tier.vram_mb == 1200

    def test_from_dict_large(self):
        data = {
            "name": "Qwen/Qwen2.5-72B-Instruct",
            "params": "72B",
            "description": "Highest quality",
            "port": 8012,
            "gpu_cost_per_hour": 3.59,
            "vram_mb": 45000,
        }
        tier = ModelTier.from_dict("large", data)
        assert tier.name == "large"
        assert tier.gpu_cost_per_hour == 3.59

    def test_from_dict_defaults(self):
        data = {"name": "some-model"}
        tier = ModelTier.from_dict("test", data)
        assert tier.port == 8010
        assert tier.gpu_cost_per_hour == 0.0
        assert tier.vram_mb == 0


class TestLoadConfig:
    def test_loads_yaml(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "cost.yaml")
        assert isinstance(cfg, CostConfig)
        assert len(cfg.models) == 3
        assert "small" in [m.name for m in cfg.models]

    def test_cascade_config(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "cost.yaml")
        assert cfg.cascade.quality_threshold == 0.8
        assert "simple" in cfg.cascade.routing
        assert cfg.cascade.routing["simple"] == "small"

    def test_benchmark_prompts(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "cost.yaml")
        assert "simple" in cfg.prompts
        assert "complex" in cfg.prompts
        assert len(cfg.prompts["simple"]) > 0

    def test_cost_analysis(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "cost.yaml")
        assert cfg.cost_analysis.gpu_hours_per_month == 720
        assert cfg.cost_analysis.target_utilization == 0.5
        assert "openai_gpt4o" in cfg.cost_analysis.api_comparison
