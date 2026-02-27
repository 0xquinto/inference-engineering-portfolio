"""
Routing strategy with override rules.

Handles explicit model requests, fallback logic when a model
is unavailable, and the default complexity-based routing.
"""

from .classifier import ComplexityClassifier


class RoutingStrategy:
    """
    Determines which model handles a request.

    Priority order:
    1. Explicit model request in the payload (user override)
    2. Complexity-based classification (default)
    3. Fallback to small model if large is unavailable
    """

    def __init__(
        self,
        classifier: ComplexityClassifier,
        available_models: list[str],
    ):
        self.classifier = classifier
        self.available_models = set(available_models)

    def select_model(
        self,
        messages: list[dict],
        requested_model: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> tuple[str, str]:
        """
        Returns (model_key, reason) tuple.

        model_key: "small" or "large"
        reason: human-readable explanation of the routing decision
        """
        # Override: explicit model request
        if requested_model:
            if requested_model in self.available_models:
                return requested_model, f"Explicitly requested: {requested_model}"
            # Map common model name patterns
            if "70b" in requested_model.lower() or "large" in requested_model.lower():
                if "large" in self.available_models:
                    return "large", f"Mapped '{requested_model}' to large model"
            if "8b" in requested_model.lower() or "small" in requested_model.lower():
                if "small" in self.available_models:
                    return "small", f"Mapped '{requested_model}' to small model"

        # Default: complexity-based routing
        result = self.classifier.classify(messages, tools, response_format)
        model_key = "large" if result.score >= self.classifier.threshold else "small"

        # Fallback: if selected model isn't available
        if model_key not in self.available_models:
            fallback = "small" if "small" in self.available_models else "large"
            return fallback, f"Fallback to {fallback} ({model_key} unavailable). {result.reason}"

        return model_key, result.reason
