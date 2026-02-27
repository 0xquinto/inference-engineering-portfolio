"""Tests for the inference engine (mocked httpx)."""

import pytest
import httpx

from src.serving.engine import InferenceEngine


@pytest.fixture
def engine():
    e = InferenceEngine()
    e.register_server("small", "http://localhost:8001")
    e.register_server("large", "http://localhost:8002")
    return e


@pytest.mark.asyncio
async def test_health_check_success(engine, monkeypatch):
    async def mock_get(self, url, **kwargs):
        return httpx.Response(200)

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    assert await engine.health_check("small") is True


@pytest.mark.asyncio
async def test_health_check_failure(engine, monkeypatch):
    async def mock_get(self, url, **kwargs):
        raise httpx.RequestError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    assert await engine.health_check("small") is False


@pytest.mark.asyncio
async def test_health_check_unregistered(engine):
    assert await engine.health_check("unknown") is False


@pytest.mark.asyncio
async def test_generate_success(engine, monkeypatch):
    async def mock_post(self, url, **kwargs):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    result = await engine.generate(
        model_key="small",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert result.text == "Hello!"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_generate_retries_without_tools_on_400(engine, monkeypatch):
    call_count = 0

    async def mock_post(self, url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                400,
                text="tool parsing error",
                request=httpx.Request("POST", url),
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Retried!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    result = await engine.generate(
        model_key="small",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[{"type": "function", "function": {"name": "f"}}],
    )
    assert result.text == "Retried!"
    assert call_count == 2


@pytest.mark.asyncio
async def test_generate_raises_on_non_tool_error(engine, monkeypatch):
    async def mock_post(self, url, **kwargs):
        return httpx.Response(
            500,
            text="internal server error",
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    with pytest.raises(httpx.HTTPStatusError):
        await engine.generate(
            model_key="small",
            messages=[{"role": "user", "content": "Hi"}],
        )


@pytest.mark.asyncio
async def test_register_server(engine):
    engine.register_server("medium", "http://localhost:9000")
    assert "medium" in engine._servers
    assert engine._servers["medium"] == "http://localhost:9000"
