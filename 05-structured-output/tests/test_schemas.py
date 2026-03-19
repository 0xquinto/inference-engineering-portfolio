import json

import pytest

from src.schemas import get_schema, get_prompt, validate_output


class TestGetSchema:
    def test_simple_json_keys(self):
        schema = get_schema("simple_json")
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert "city" in schema["properties"]

    def test_nested_object_keys(self):
        schema = get_schema("nested_object")
        assert isinstance(schema, dict)
        assert "user" in schema["properties"]
        assert "orders" in schema["properties"]

    def test_function_call_keys(self):
        schema = get_schema("function_call")
        assert isinstance(schema, dict)
        assert "function" in schema["properties"]
        assert "arguments" in schema["properties"]
        assert "enum" in schema["properties"]["function"]

    def test_unknown_schema_raises(self):
        with pytest.raises(ValueError):
            get_schema("nonexistent")


class TestGetPrompt:
    def test_simple_json_prompt(self):
        prompt = get_prompt("simple_json")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_nested_object_prompt(self):
        prompt = get_prompt("nested_object")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_function_call_prompt(self):
        prompt = get_prompt("function_call")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_unknown_prompt_raises(self):
        with pytest.raises(ValueError):
            get_prompt("nonexistent")


class TestValidateOutput:
    def test_valid_simple_json(self):
        output = json.dumps({"name": "Maria", "age": 34, "city": "Barcelona"})
        assert validate_output("simple_json", output) is True

    def test_valid_nested_object(self):
        output = json.dumps({
            "user": {
                "name": "Alex",
                "address": {"street": "5th Ave", "city": "New York", "zip": "10001"},
            },
            "orders": [
                {"id": 1, "item": "Widget", "price": 9.99},
            ],
        })
        assert validate_output("nested_object", output) is True

    def test_valid_function_call(self):
        output = json.dumps({
            "function": "search",
            "arguments": {"query": "weather in Paris"},
        })
        assert validate_output("function_call", output) is True

    def test_invalid_json_string(self):
        assert validate_output("simple_json", "not json at all") is False

    def test_missing_required_field(self):
        output = json.dumps({"name": "Maria", "city": "Barcelona"})
        assert validate_output("simple_json", output) is False

    def test_wrong_type(self):
        output = json.dumps({"name": "Maria", "age": "thirty", "city": "Barcelona"})
        assert validate_output("simple_json", output) is False

    def test_invalid_enum_value(self):
        output = json.dumps({
            "function": "invalid_func",
            "arguments": {},
        })
        assert validate_output("function_call", output) is False

    def test_none_input(self):
        assert validate_output("simple_json", None) is False
