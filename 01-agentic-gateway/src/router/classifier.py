"""
Request complexity classifier.

Scores incoming requests on a 0-1 scale to determine routing.
Starts with fast heuristics, can be upgraded to a learned classifier later.
"""

from dataclasses import dataclass

import yaml


HIGH_COMPLEXITY_KEYWORDS = [
    "analyze", "compare", "explain in detail", "write code", "debug",
    "refactor", "multi-step", "implement", "design", "architect",
    "optimize", "evaluate", "critique", "step by step", "reasoning",
]


@dataclass
class ComplexityScore:
    score: float          # 0.0 (simple) to 1.0 (complex)
    reason: str           # human-readable explanation
    signals: dict         # individual signal scores for debugging


class ComplexityClassifier:
    """
    Heuristic-based complexity classifier.

    Scoring signals:
    1. Message length (longer = more complex)
    2. Keyword detection (certain words indicate complex tasks)
    3. Tool use (if tools are provided, likely needs the stronger model)
    4. Message count (multi-turn conversations are often more complex)
    5. System prompt complexity (long system prompts = specialized task)
    """

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def classify(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> ComplexityScore:
        signals = {}

        # Signal 1: Last user message length
        last_user_msg = self._get_last_user_message(messages)
        msg_len = len(last_user_msg.split())
        if msg_len < 20:
            signals["message_length"] = 0.1
        elif msg_len < 50:
            signals["message_length"] = 0.3
        elif msg_len < 150:
            signals["message_length"] = 0.6
        else:
            signals["message_length"] = 0.9

        # Signal 2: Keyword complexity
        lower_msg = last_user_msg.lower()
        keyword_hits = sum(1 for kw in HIGH_COMPLEXITY_KEYWORDS if kw in lower_msg)
        signals["keyword_complexity"] = min(keyword_hits * 0.25, 1.0)

        # Signal 3: Tool use requested
        signals["tool_use"] = 0.8 if tools and len(tools) > 0 else 0.0

        # Signal 4: Conversation depth
        user_msg_count = sum(1 for m in messages if m.get("role") == "user")
        signals["conversation_depth"] = min(user_msg_count * 0.15, 0.6)

        # Signal 5: Structured output requested
        signals["structured_output"] = 0.7 if response_format else 0.0

        # Weighted average
        weights = {
            "message_length": 0.2,
            "keyword_complexity": 0.35,
            "tool_use": 0.2,
            "conversation_depth": 0.1,
            "structured_output": 0.15,
        }
        score = sum(signals[k] * weights[k] for k in signals)
        score = max(0.0, min(1.0, score))

        # Build reason
        top_signal = max(signals, key=signals.get)
        if score >= self.threshold:
            reason = f"Complex (score={score:.2f}, top signal: {top_signal}={signals[top_signal]:.2f})"
        else:
            reason = f"Simple (score={score:.2f}, top signal: {top_signal}={signals[top_signal]:.2f})"

        return ComplexityScore(score=score, reason=reason, signals=signals)

    def route(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Returns 'small' or 'large' model key."""
        result = self.classify(messages, tools, response_format)
        return "large" if result.score >= self.threshold else "small"

    def _get_last_user_message(self, messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle multimodal messages
                    return " ".join(
                        part.get("text", "") for part in content if part.get("type") == "text"
                    )
                return content
        return ""
