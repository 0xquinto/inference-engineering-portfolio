import pytest

from src.cascade import classify_complexity, CascadeDecision, CascadeRouter
from src.config import CascadeConfig, ModelTier


@pytest.fixture
def keywords():
    return {
        "simple": ["what is", "who is", "define", "capital of"],
        "moderate": ["explain", "compare", "summarize", "how does"],
        "complex": ["analyze", "design", "implement", "write a function"],
    }


@pytest.fixture
def cascade_config(keywords):
    return CascadeConfig(
        complexity_keywords=keywords,
        routing={"simple": "small", "moderate": "medium", "complex": "large"},
        quality_threshold=0.8,
    )


@pytest.fixture
def model_tiers():
    return [
        ModelTier("small", "Qwen/Qwen3.5-0.8B", "0.8B", "Small", 8010, 0.75, 1200),
        ModelTier("medium", "Qwen/Qwen3.5-9B", "9B", "Medium", 8011, 0.75, 14200),
        ModelTier("large", "Qwen/Qwen3.5-27B", "27B", "Large", 8012, 3.59, 28000),
    ]


class TestClassifyComplexity:
    def test_simple_prompt(self, keywords):
        assert classify_complexity("What is the capital of France?", keywords) == "simple"

    def test_moderate_prompt(self, keywords):
        assert classify_complexity("Explain the difference between TCP and UDP.", keywords) == "moderate"

    def test_complex_prompt(self, keywords):
        assert classify_complexity("Design a distributed cache strategy.", keywords) == "complex"

    def test_case_insensitive(self, keywords):
        assert classify_complexity("WHAT IS photosynthesis?", keywords) == "simple"

    def test_defaults_to_complex(self, keywords):
        assert classify_complexity("Tell me a joke about cats.", keywords) == "complex"

    def test_empty_prompt(self, keywords):
        assert classify_complexity("", keywords) == "complex"


class TestCascadeRouter:
    def test_route_simple(self, cascade_config, model_tiers):
        router = CascadeRouter(cascade_config, model_tiers)
        decision = router.route("What is the capital of France?")
        assert isinstance(decision, CascadeDecision)
        assert decision.classified_complexity == "simple"
        assert decision.routed_to == "small"
        assert decision.final_model == "small"
        assert decision.escalated is False

    def test_route_moderate(self, cascade_config, model_tiers):
        router = CascadeRouter(cascade_config, model_tiers)
        decision = router.route("Explain quantum computing.")
        assert decision.classified_complexity == "moderate"
        assert decision.routed_to == "medium"

    def test_route_complex(self, cascade_config, model_tiers):
        router = CascadeRouter(cascade_config, model_tiers)
        decision = router.route("Design a microservices architecture.")
        assert decision.classified_complexity == "complex"
        assert decision.routed_to == "large"

    def test_route_unknown_defaults_large(self, cascade_config, model_tiers):
        router = CascadeRouter(cascade_config, model_tiers)
        decision = router.route("Random unmatched prompt here.")
        assert decision.routed_to == "large"

    def test_should_escalate_below_threshold(self, cascade_config, model_tiers):
        router = CascadeRouter(cascade_config, model_tiers)
        assert router.should_escalate(0.5) is True

    def test_should_not_escalate_above_threshold(self, cascade_config, model_tiers):
        router = CascadeRouter(cascade_config, model_tiers)
        assert router.should_escalate(0.9) is False

    def test_should_escalate_at_threshold(self, cascade_config, model_tiers):
        router = CascadeRouter(cascade_config, model_tiers)
        assert router.should_escalate(0.8) is False
