"""Tests for the complexity classifier and routing strategy."""

from src.router.classifier import ComplexityClassifier


def test_simple_message_routes_to_small():
    classifier = ComplexityClassifier(threshold=0.6)
    messages = [{"role": "user", "content": "What is 2+2?"}]
    result = classifier.route(messages)
    assert result == "small"


def test_complex_message_routes_to_large():
    classifier = ComplexityClassifier(threshold=0.6)
    messages = [{"role": "user", "content": "Analyze and compare the performance characteristics of quicksort versus mergesort. Write code to implement both and debug any issues."}]
    result = classifier.route(messages)
    assert result == "large"


def test_tool_use_increases_complexity():
    classifier = ComplexityClassifier(threshold=0.6)
    messages = [{"role": "user", "content": "Get the weather"}]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    score_without = classifier.classify(messages).score
    score_with = classifier.classify(messages, tools=tools).score

    assert score_with > score_without


def test_structured_output_increases_complexity():
    classifier = ComplexityClassifier(threshold=0.6)
    messages = [{"role": "user", "content": "List three colors"}]

    score_without = classifier.classify(messages).score
    score_with = classifier.classify(messages, response_format={"type": "json_object"}).score

    assert score_with > score_without


def test_complexity_score_in_range():
    classifier = ComplexityClassifier(threshold=0.6)
    messages = [{"role": "user", "content": "Hello"}]
    result = classifier.classify(messages)
    assert 0.0 <= result.score <= 1.0


def test_multimodal_message_handled():
    classifier = ComplexityClassifier(threshold=0.6)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ],
    }]
    result = classifier.classify(messages)
    assert result.score >= 0.0
