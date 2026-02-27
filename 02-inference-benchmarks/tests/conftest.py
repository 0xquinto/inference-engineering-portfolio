"""Shared test fixtures."""

import pytest
from src.runners.base import RequestConfig


@pytest.fixture
def request_config():
    return RequestConfig(max_tokens=64, temperature=0.0)
