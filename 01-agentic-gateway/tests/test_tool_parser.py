"""Tests for the tool call parser."""

from src.agents.tool_parser import ToolCallParser


def test_parse_native_tool_calls(tool_parser):
    response = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "London"}',
                    },
                }],
            },
        }],
    }
    result = tool_parser.parse_complete_response(response)
    assert result.has_tool_calls
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function_name == "get_weather"
    assert result.tool_calls[0].parsed_arguments == {"city": "London"}


def test_parse_multiple_tool_calls(tool_parser):
    response = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "search", "arguments": '{"q": "test"}'}},
                    {"id": "call_2", "function": {"name": "fetch", "arguments": '{"url": "http://x.com"}'}},
                ],
            },
        }],
    }
    result = tool_parser.parse_complete_response(response)
    assert len(result.tool_calls) == 2
    assert result.tool_calls[1].function_name == "fetch"


def test_json_fallback_in_content(tool_parser):
    response = {
        "choices": [{
            "message": {
                "content": 'I will call {"name": "get_weather", "arguments": {"city": "Paris"}} now.',
                "tool_calls": None,
            },
        }],
    }
    result = tool_parser.parse_complete_response(response)
    assert result.has_tool_calls
    assert result.tool_calls[0].function_name == "get_weather"


def test_no_tool_calls(tool_parser):
    response = {
        "choices": [{
            "message": {
                "content": "Just a plain text response.",
            },
        }],
    }
    result = tool_parser.parse_complete_response(response)
    assert not result.has_tool_calls
    assert result.content == "Just a plain text response."


def test_streaming_deltas(tool_parser):
    deltas = [
        {"content": "Hello"},
        {"content": " world"},
        {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "search", "arguments": ""}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '{"q":'}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": ' "test"}'}}]},
    ]
    result = tool_parser.parse_streaming_chunks(deltas)
    assert result.content == "Hello world"
    assert result.has_tool_calls
    assert result.tool_calls[0].function_name == "search"
    assert result.tool_calls[0].parsed_arguments == {"q": "test"}


def test_streaming_multiple_tool_calls(tool_parser):
    deltas = [
        {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "a", "arguments": "{}"}}]},
        {"tool_calls": [{"index": 1, "id": "call_2", "function": {"name": "b", "arguments": "{}"}}]},
    ]
    result = tool_parser.parse_streaming_chunks(deltas)
    assert len(result.tool_calls) == 2


def test_safe_parse_json_invalid(tool_parser):
    assert tool_parser._safe_parse_json("not json") is None
    assert tool_parser._safe_parse_json("") is None
