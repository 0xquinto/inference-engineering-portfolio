"""Tests for the complexity classifier and routing strategy."""

from src.router.classifier import ComplexityClassifier


def test_simple_message_routes_to_small():
    classifier = ComplexityClassifier(threshold=0.6)
    messages = [{"role": "user", "content": "What is 2+2?"}]
    result = classifier.route(messages)
    assert result == "small"


def test_complex_message_routes_to_large():
    classifier = ComplexityClassifier(threshold=0.6)
    messages = [{"role": "user", "content": (
        "Analyze and compare the performance characteristics of quicksort versus mergesort "
        "across different input distributions including sorted, reverse sorted, and random data. "
        "Write code to implement both algorithms and debug any issues you find. "
        "Evaluate which is more suitable for large-scale production workloads."
    )}]
    tools = [{"type": "function", "function": {"name": "run_code"}}]
    result = classifier.route(messages, tools=tools, response_format={"type": "json_object"})
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


def test_conversation_depth_increases_score():
    classifier = ComplexityClassifier(threshold=0.6)
    single = [{"role": "user", "content": "Hello"}]
    multi = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Follow up"},
        {"role": "assistant", "content": "Sure"},
        {"role": "user", "content": "Another question"},
    ]
    score_single = classifier.classify(single).score
    score_multi = classifier.classify(multi).score
    assert score_multi > score_single


def test_threshold_boundary():
    classifier = ComplexityClassifier(threshold=0.6)
    # A message designed to land right near the threshold
    messages = [{"role": "user", "content": "Analyze this data"}]
    result = classifier.classify(messages)
    if result.score >= 0.6:
        assert classifier.route(messages) == "large"
    else:
        assert classifier.route(messages) == "small"


def test_empty_messages_handled():
    classifier = ComplexityClassifier(threshold=0.6)
    result = classifier.classify([])
    assert 0.0 <= result.score <= 1.0
    assert classifier.route([]) == "small"
