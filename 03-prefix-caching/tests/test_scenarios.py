import pytest

from src.scenarios import (
    generate_shared_system_prompt,
    generate_multi_turn,
    generate_rag_context,
    generate_cache_pressure,
)


class TestSharedSystemPrompt:
    def test_all_share_same_system(self):
        requests = generate_shared_system_prompt(num_requests=10, system_tokens=100, user_tokens=20)
        system_msgs = [r[0] for r in requests]
        assert all(s == system_msgs[0] for s in system_msgs)
        assert len(requests) == 10

    def test_user_messages_vary(self):
        requests = generate_shared_system_prompt(num_requests=10, system_tokens=100, user_tokens=20)
        user_msgs = [r[1] for r in requests]
        assert len(set(m["content"] for m in user_msgs)) > 1

    def test_message_format(self):
        requests = generate_shared_system_prompt(num_requests=1, system_tokens=50, user_tokens=20)
        system, user = requests[0]
        assert system["role"] == "system"
        assert user["role"] == "user"


class TestMultiTurn:
    def test_turn_count(self):
        convos = generate_multi_turn(num_conversations=2, turns=5, tokens_per_turn=30)
        assert len(convos) == 2
        assert len(convos[0]) == 5

    def test_growing_history(self):
        convos = generate_multi_turn(num_conversations=1, turns=5, tokens_per_turn=30)
        for i, turn_messages in enumerate(convos[0]):
            # Each turn has all previous messages + new user message
            user_msgs = [m for m in turn_messages if m["role"] == "user"]
            assert len(user_msgs) == i + 1


class TestRAGContext:
    def test_shared_context(self):
        requests = generate_rag_context(context_tokens=200, num_queries=5, query_tokens=20)
        contexts = [r[0]["content"] for r in requests]
        assert all(c == contexts[0] for c in contexts)
        assert len(requests) == 5

    def test_different_queries(self):
        requests = generate_rag_context(context_tokens=200, num_queries=5, query_tokens=20)
        queries = [r[1]["content"] for r in requests]
        assert len(set(queries)) > 1


class TestCachePressure:
    def test_unique_prefix_count(self):
        batches = generate_cache_pressure(
            prefix_tokens=50, unique_prefixes=[5, 10], requests_per_prefix=3
        )
        assert len(batches) == 2
        assert len(batches[0]) == 15  # 5 prefixes * 3 requests
        assert len(batches[1]) == 30  # 10 prefixes * 3 requests
