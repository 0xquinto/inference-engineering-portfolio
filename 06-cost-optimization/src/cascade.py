from dataclasses import dataclass

from .config import CascadeConfig, ModelTier


def classify_complexity(prompt: str, keywords: dict[str, list[str]]) -> str:
    """Classify prompt complexity based on keyword matching.

    Checks in order: simple, moderate, complex. Returns the first match.
    Defaults to "complex" if no keywords match.
    """
    prompt_lower = prompt.lower()
    for level in ("simple", "moderate", "complex"):
        for keyword in keywords.get(level, []):
            if keyword.lower() in prompt_lower:
                return level
    return "complex"


@dataclass
class CascadeDecision:
    prompt: str
    classified_complexity: str
    routed_to: str
    escalated: bool
    final_model: str


class CascadeRouter:
    def __init__(self, config: CascadeConfig, models: list[ModelTier]):
        self.config = config
        self.models = {m.name: m for m in models}

    def route(self, prompt: str) -> CascadeDecision:
        complexity = classify_complexity(prompt, self.config.complexity_keywords)
        tier_name = self.config.routing.get(complexity, "large")
        return CascadeDecision(
            prompt=prompt,
            classified_complexity=complexity,
            routed_to=tier_name,
            escalated=False,
            final_model=tier_name,
        )

    def should_escalate(self, quality_score: float) -> bool:
        return quality_score < self.config.quality_threshold
