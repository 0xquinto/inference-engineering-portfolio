import pytest
from unittest.mock import patch, MagicMock

from src.evaluate import EvalResult, QualityEvaluator


class TestEvalResult:
    def test_delta_from_baseline(self):
        baseline = EvalResult(format_name="bf16", perplexity=6.82, mmlu_accuracy=0.714)
        quantized = EvalResult(format_name="gptq_int4", perplexity=7.01, mmlu_accuracy=0.698)
        delta = quantized.delta_from(baseline)
        assert delta["perplexity_delta"] == pytest.approx(0.19, abs=0.01)
        assert delta["mmlu_delta"] == pytest.approx(-0.016, abs=0.001)

    def test_delta_percentage(self):
        baseline = EvalResult(format_name="bf16", perplexity=6.82, mmlu_accuracy=0.714)
        quantized = EvalResult(format_name="gptq_int4", perplexity=7.01, mmlu_accuracy=0.698)
        delta = quantized.delta_from(baseline)
        assert delta["perplexity_pct"] == pytest.approx(2.79, abs=0.1)
        assert delta["mmlu_pct"] == pytest.approx(-2.24, abs=0.1)

    def test_to_dict(self):
        r = EvalResult(format_name="bf16", perplexity=6.82, mmlu_accuracy=0.714)
        d = r.to_dict()
        assert d["format_name"] == "bf16"
        assert d["perplexity"] == 6.82


class TestQualityEvaluator:
    def test_init(self):
        evaluator = QualityEvaluator(model_path="test/model", max_samples=100)
        assert evaluator.model_path == "test/model"
        assert evaluator.max_samples == 100
