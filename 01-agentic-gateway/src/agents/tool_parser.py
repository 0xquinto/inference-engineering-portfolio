"""
Tool call parser for streaming output.

Detects and extracts function/tool calls from streaming LLM output.
Handles both OpenAI-format tool calls and raw JSON function calls.
"""

import json
import re
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    function_name: str
    arguments: str  # raw JSON string
    parsed_arguments: dict | None = None


@dataclass
class ParsedResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    has_tool_calls: bool = False


class ToolCallParser:
    """
    Parses tool calls from model output.

    Handles two formats:
    1. OpenAI-native tool calls (via vLLM's tool calling support)
    2. Raw JSON tool calls embedded in text (fallback for models without native support)
    """

    def parse_complete_response(self, response_data: dict) -> ParsedResponse:
        """Parse a complete (non-streaming) response for tool calls."""
        choice = response_data["choices"][0]
        message = choice["message"]

        content = message.get("content") or ""
        tool_calls = []

        # Check for native tool calls
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    function_name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                    parsed_arguments=self._safe_parse_json(tc["function"]["arguments"]),
                ))

        # Fallback: check for JSON tool calls in content
        if not tool_calls and content:
            extracted = self._extract_json_tool_calls(content)
            tool_calls.extend(extracted)

        return ParsedResponse(
            content=content,
            tool_calls=tool_calls,
            has_tool_calls=len(tool_calls) > 0,
        )

    def parse_streaming_chunks(self, accumulated_deltas: list[dict]) -> ParsedResponse:
        """
        Parse accumulated streaming deltas for tool calls.
        Call this after the stream completes.
        """
        content_parts = []
        tool_call_parts: dict[int, dict] = {}  # index -> {id, name, args}

        for delta in accumulated_deltas:
            if delta.get("content"):
                content_parts.append(delta["content"])

            if delta.get("tool_calls"):
                for tc_delta in delta["tool_calls"]:
                    idx = tc_delta["index"]
                    if idx not in tool_call_parts:
                        tool_call_parts[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.get("id"):
                        tool_call_parts[idx]["id"] = tc_delta["id"]
                    if tc_delta.get("function", {}).get("name"):
                        tool_call_parts[idx]["name"] = tc_delta["function"]["name"]
                    if tc_delta.get("function", {}).get("arguments"):
                        tool_call_parts[idx]["args"] += tc_delta["function"]["arguments"]

        content = "".join(content_parts)
        tool_calls = []
        for idx in sorted(tool_call_parts.keys()):
            tc = tool_call_parts[idx]
            tool_calls.append(ToolCall(
                id=tc["id"],
                function_name=tc["name"],
                arguments=tc["args"],
                parsed_arguments=self._safe_parse_json(tc["args"]),
            ))

        return ParsedResponse(
            content=content,
            tool_calls=tool_calls,
            has_tool_calls=len(tool_calls) > 0,
        )

    def _extract_json_tool_calls(self, text: str) -> list[ToolCall]:
        """Fallback: extract tool calls from raw text containing JSON."""
        tool_calls = []
        # Look for patterns like {"name": "func", "arguments": {...}}
        pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}'
        for i, match in enumerate(re.finditer(pattern, text)):
            tool_calls.append(ToolCall(
                id=f"call_{i}",
                function_name=match.group(1),
                arguments=match.group(2),
                parsed_arguments=self._safe_parse_json(match.group(2)),
            ))
        return tool_calls

    def _safe_parse_json(self, s: str) -> dict | None:
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return None
