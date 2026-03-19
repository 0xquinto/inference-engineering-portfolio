from pathlib import Path

from src.config import SpecMethod, SpecConfig, load_config


class TestSpecMethod:
    def test_from_dict_baseline(self):
        data = {"description": "Autoregressive decoding (no speculation)", "spec_type": None}
        method = SpecMethod.from_dict("baseline", data)
        assert method.name == "baseline"
        assert method.is_baseline
        assert method.spec_type is None

    def test_from_dict_eagle3(self):
        data = {
            "description": "EAGLE-3 hidden-state speculation",
            "spec_type": "eagle",
            "draft_model": None,
            "num_speculative_tokens": 5,
        }
        method = SpecMethod.from_dict("eagle3", data)
        assert method.name == "eagle3"
        assert not method.is_baseline
        assert method.spec_type == "eagle"
        assert method.num_speculative_tokens == 5

    def test_from_dict_ngram(self):
        data = {
            "description": "N-gram suffix matching",
            "spec_type": "ngram",
            "ngram_prompt_lookup_max": 5,
            "ngram_prompt_lookup_min": 2,
            "num_speculative_tokens": 5,
        }
        method = SpecMethod.from_dict("ngram", data)
        assert method.spec_type == "ngram"
        assert method.ngram_prompt_lookup_max == 5
        assert method.ngram_prompt_lookup_min == 2

    def test_from_dict_draft_model(self):
        data = {
            "description": "Small draft model verification",
            "spec_type": "draft_model",
            "draft_model": "Qwen/Qwen3.5-0.8B",
            "num_speculative_tokens": 5,
        }
        method = SpecMethod.from_dict("draft_model", data)
        assert method.spec_type == "draft_model"
        assert method.draft_model == "Qwen/Qwen3.5-0.8B"

    def test_is_baseline_true(self):
        method = SpecMethod.from_dict("baseline", {"description": "baseline", "spec_type": None})
        assert method.is_baseline is True

    def test_is_baseline_false(self):
        method = SpecMethod.from_dict("eagle3", {"description": "eagle", "spec_type": "eagle"})
        assert method.is_baseline is False

    def test_vllm_args_baseline(self):
        method = SpecMethod.from_dict("baseline", {"description": "baseline", "spec_type": None})
        assert method.vllm_args() == []

    def test_vllm_args_eagle(self):
        method = SpecMethod.from_dict("eagle3", {
            "description": "eagle",
            "spec_type": "eagle",
            "num_speculative_tokens": 5,
        })
        args = method.vllm_args()
        assert "--speculative-model" in args
        assert "[eagle_head]" in args
        assert "--num-speculative-tokens" in args
        assert "5" in args

    def test_vllm_args_eagle_parallel(self):
        method = SpecMethod.from_dict("p_eagle", {
            "description": "parallel eagle",
            "spec_type": "eagle",
            "num_speculative_tokens": 5,
            "parallel_drafting": True,
        })
        args = method.vllm_args()
        assert "--speculative-config" in args

    def test_vllm_args_ngram(self):
        method = SpecMethod.from_dict("ngram", {
            "description": "ngram",
            "spec_type": "ngram",
            "ngram_prompt_lookup_max": 5,
            "ngram_prompt_lookup_min": 2,
            "num_speculative_tokens": 5,
        })
        args = method.vllm_args()
        assert "--speculative-model" in args
        assert "[ngram]" in args
        assert "--ngram-prompt-lookup-max" in args
        assert "5" in args
        assert "--ngram-prompt-lookup-min" in args
        assert "2" in args

    def test_vllm_args_mtp(self):
        method = SpecMethod.from_dict("mtp", {
            "description": "Multi-Token Prediction",
            "spec_type": "mtp",
            "num_speculative_tokens": 1,
        })
        args = method.vllm_args()
        assert "--speculative-model" in args
        assert "[mtp]" in args
        assert "--num-speculative-tokens" in args
        assert "1" in args

    def test_vllm_args_draft_model(self):
        method = SpecMethod.from_dict("draft_model", {
            "description": "draft",
            "spec_type": "draft_model",
            "draft_model": "Qwen/Qwen3.5-0.8B",
            "num_speculative_tokens": 5,
        })
        args = method.vllm_args()
        assert "--speculative-model" in args
        assert "Qwen/Qwen3.5-0.8B" in args
        assert "--num-speculative-tokens" in args
        assert "5" in args


class TestLoadConfig:
    def test_loads_yaml(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "speculative.yaml")
        assert isinstance(cfg, SpecConfig)
        assert cfg.model_name == "Qwen/Qwen3.5-9B"
        assert len(cfg.methods) == 6
        assert "baseline" in [m.name for m in cfg.methods]

    def test_benchmark_prompts(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "speculative.yaml")
        assert len(cfg.benchmark_prompts) > 0

    def test_qps_levels(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "speculative.yaml")
        assert cfg.qps_levels == [1, 5, 10, 25, 50]

    def test_defaults(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "speculative.yaml")
        assert cfg.port == 8010
        assert cfg.max_tokens == 256
        assert cfg.temperature == 0.0
