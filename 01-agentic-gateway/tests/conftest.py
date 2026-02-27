"""Shared fixtures for the test suite."""

import pytest

from src.router.classifier import ComplexityClassifier
from src.agents.tool_parser import ToolCallParser


@pytest.fixture
def classifier():
    return ComplexityClassifier(threshold=0.6)


@pytest.fixture
def tool_parser():
    return ToolCallParser()


@pytest.fixture
def simple_messages():
    return [{"role": "user", "content": "What is 2+2?"}]


@pytest.fixture
def complex_messages():
    return [
        {"role": "user", "content": "Analyze and compare quicksort vs mergesort. Write code to implement both and debug any issues."},
    ]


@pytest.fixture
def multi_turn_messages():
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Tell me about Python"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Can you explain decorators?"},
    ]


@pytest.fixture
def sample_tools():
    return [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}},
        {"type": "function", "function": {"name": "search_web", "parameters": {}}},
    ]
