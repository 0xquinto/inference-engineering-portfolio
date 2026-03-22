from pathlib import Path

from src.config import Backend, SchemaLevel, StructuredConfig, load_config


class TestBackend:
    def test_from_dict_xgrammar(self):
        data = {
            "description": "XGrammar guided decoding",
            "guided_decoding_backend": "xgrammar",
        }
        b = Backend.from_dict("xgrammar", data)
        assert b.name == "xgrammar"
        assert b.guided_decoding_backend == "xgrammar"
        assert b.description == "XGrammar guided decoding"

    def test_from_dict_unconstrained(self):
        data = {
            "description": "No grammar enforcement",
            "guided_decoding_backend": None,
        }
        b = Backend.from_dict("unconstrained", data)
        assert b.name == "unconstrained"
        assert b.guided_decoding_backend is None

    def test_is_constrained_true(self):
        b = Backend.from_dict("xgrammar", {
            "description": "XGrammar",
            "guided_decoding_backend": "xgrammar",
        })
        assert b.is_constrained is True

    def test_is_constrained_false(self):
        b = Backend.from_dict("unconstrained", {
            "description": "Unconstrained",
            "guided_decoding_backend": None,
        })
        assert b.is_constrained is False


class TestSchemaLevel:
    def test_from_dict(self):
        data = {
            "description": "Flat JSON object",
            "complexity": "low",
        }
        s = SchemaLevel.from_dict("simple_json", data)
        assert s.name == "simple_json"
        assert s.complexity == "low"
        assert s.description == "Flat JSON object"


class TestLoadConfig:
    def test_loads_yaml(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "structured.yaml")
        assert isinstance(cfg, StructuredConfig)
        assert cfg.model_name == "Qwen/Qwen3.5-9B"
        assert len(cfg.backends) == 3
        assert "xgrammar" in [b.name for b in cfg.backends]

    def test_schema_levels(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "structured.yaml")
        assert len(cfg.schemas) == 3
        names = [s.name for s in cfg.schemas]
        assert "simple_json" in names
        assert "nested_object" in names
        assert "function_call" in names

    def test_concurrency_levels(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "structured.yaml")
        assert cfg.concurrency_levels == [1, 10, 50]

    def test_max_retries(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "structured.yaml")
        assert cfg.max_retries == 3

    def test_disable_thinking_default(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "structured.yaml")
        assert cfg.disable_thinking is False

    def test_disable_thinking_gpu_profile(self):
        cfg = load_config(Path(__file__).parent.parent / "profiles" / "gpu.yaml")
        assert cfg.disable_thinking is True
