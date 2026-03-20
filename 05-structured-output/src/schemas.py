import json
import re

SCHEMAS = {
    "simple_json": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"},
        },
        "required": ["name", "age", "city"],
        "additionalProperties": False,
    },
    "nested_object": {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                            "zip": {"type": "string"},
                        },
                        "required": ["street", "city", "zip"],
                    },
                },
                "required": ["name", "address"],
            },
            "orders": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "item": {"type": "string"},
                        "price": {"type": "number"},
                    },
                    "required": ["id", "item", "price"],
                },
            },
        },
        "required": ["user", "orders"],
        "additionalProperties": False,
    },
    "function_call": {
        "type": "object",
        "properties": {
            "function": {
                "type": "string",
                "enum": ["search", "calculate", "send_email"],
            },
            "arguments": {
                "type": "object",
            },
        },
        "required": ["function", "arguments"],
        "additionalProperties": False,
    },
}

PROMPTS = {
    "simple_json": (
        "Extract the following person's information as JSON with keys "
        '"name" (string), "age" (integer), and "city" (string): '
        "Maria Garcia is a 34-year-old software engineer living in Barcelona."
    ),
    "nested_object": (
        "Generate mock e-commerce data as JSON with a 'user' object "
        "(containing 'name' and 'address' with 'street', 'city', 'zip') "
        "and an 'orders' array (each order has 'id', 'item', 'price'). "
        "Include 2 orders for a user named Alex in New York."
    ),
    "function_call": (
        "You are a function-calling assistant. Given the user query, respond "
        'with JSON containing "function" (one of: "search", "calculate", '
        '"send_email") and "arguments" (a dict of parameters for that function). '
        "User query: What is the weather in Paris right now?"
    ),
}


def get_schema(level_name: str) -> dict:
    """Return the JSON Schema dict for a given complexity level."""
    if level_name not in SCHEMAS:
        raise ValueError(f"Unknown schema level: {level_name}")
    return SCHEMAS[level_name]


def get_prompt(level_name: str) -> str:
    """Return the prompt string for a given complexity level."""
    if level_name not in PROMPTS:
        raise ValueError(f"Unknown schema level: {level_name}")
    return PROMPTS[level_name]


def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> tags from model output (e.g. Qwen3.5)."""
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def validate_output(level_name: str, output: str) -> bool:
    """Validate that output conforms to the schema for the given level."""
    output = strip_think_tags(output)
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return False

    schema = get_schema(level_name)
    return _validate_against_schema(data, schema)


def _validate_against_schema(data: object, schema: dict) -> bool:
    """Simple recursive JSON Schema validator (subset of JSON Schema)."""
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            return False
        required = schema.get("required", [])
        for key in required:
            if key not in data:
                return False
        properties = schema.get("properties", {})
        for key, prop_schema in properties.items():
            if key in data:
                if not _validate_against_schema(data[key], prop_schema):
                    return False
        return True

    elif schema_type == "array":
        if not isinstance(data, list):
            return False
        items_schema = schema.get("items", {})
        for item in data:
            if not _validate_against_schema(item, items_schema):
                return False
        return True

    elif schema_type == "string":
        if not isinstance(data, str):
            return False
        enum = schema.get("enum")
        if enum is not None and data not in enum:
            return False
        return True

    elif schema_type == "integer":
        return isinstance(data, int) and not isinstance(data, bool)

    elif schema_type == "number":
        return isinstance(data, (int, float)) and not isinstance(data, bool)

    return True
