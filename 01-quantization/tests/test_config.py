from pathlib import Path

from src.config import QuantFormat, QuantConfig, load_config


class TestQuantFormat:
    def test_from_dict_gptq(self):
        data = {
            "description": "GPTQ INT4",
            "tool": "auto_gptq",
            "bits": 4,
            "group_size": 128,
            "calibration_samples": 128,
        }
        fmt = QuantFormat.from_dict("gptq_int4", data)
        assert fmt.name == "gptq_int4"
        assert fmt.tool == "auto_gptq"
        assert fmt.bits == 4

    def test_from_dict_baseline(self):
        data = {"description": "BF16 baseline", "tool": None}
        fmt = QuantFormat.from_dict("bf16", data)
        assert fmt.is_baseline

    def test_vllm_model_id_baseline(self):
        fmt = QuantFormat.from_dict("bf16", {"description": "baseline", "tool": None})
        assert fmt.vllm_model_path("Qwen/Qwen2.5-7B-Instruct") == "Qwen/Qwen2.5-7B-Instruct"

    def test_vllm_model_id_quantized(self):
        fmt = QuantFormat.from_dict("gptq_int4", {"description": "GPTQ", "tool": "auto_gptq", "bits": 4})
        path = fmt.vllm_model_path("Qwen/Qwen2.5-7B-Instruct")
        assert path == "quantized_models/Qwen2.5-7B-Instruct-gptq_int4"


class TestLoadConfig:
    def test_loads_yaml(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "quantization.yaml")
        assert isinstance(cfg, QuantConfig)
        assert cfg.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert len(cfg.formats) == 3
        assert "bf16" in [f.name for f in cfg.formats]

    def test_benchmark_prompts(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "quantization.yaml")
        assert len(cfg.benchmark_prompts) > 0

    def test_concurrency_levels(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "quantization.yaml")
        assert cfg.concurrency_levels == [1, 10, 50]
