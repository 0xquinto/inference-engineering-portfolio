import hashlib


# Deterministic filler text generation (no external deps)
_FILLER_WORDS = (
    "the quick brown fox jumps over the lazy dog while the cat sleeps "
    "on a warm sunny afternoon near the old oak tree by the river bank "
    "where fish swim upstream against the gentle current of clear water "
    "flowing down from the snow capped mountains in the distance beyond "
)


def _filler_text(seed: int, approx_tokens: int) -> str:
    """Generate deterministic filler text of approximately `approx_tokens` words."""
    words = _FILLER_WORDS.split()
    h = hashlib.md5(str(seed).encode()).hexdigest()
    offset = int(h[:8], 16) % len(words)
    result = []
    for i in range(approx_tokens):
        result.append(words[(offset + i) % len(words)])
    return " ".join(result)


def generate_shared_system_prompt(
    num_requests: int, system_tokens: int, user_tokens: int
) -> list[tuple[dict, dict]]:
    system_msg = {
        "role": "system",
        "content": f"You are a helpful assistant. {_filler_text(seed=0, approx_tokens=system_tokens)}",
    }
    requests = []
    for i in range(num_requests):
        user_msg = {
            "role": "user",
            "content": f"Question {i}: {_filler_text(seed=1000 + i, approx_tokens=user_tokens)}",
        }
        requests.append((system_msg, user_msg))
    return requests


def generate_multi_turn(
    num_conversations: int, turns: int, tokens_per_turn: int
) -> list[list[list[dict]]]:
    conversations = []
    for c in range(num_conversations):
        history = []
        turn_snapshots = []
        for t in range(turns):
            user_msg = {
                "role": "user",
                "content": f"Turn {t}: {_filler_text(seed=2000 + c * 100 + t, approx_tokens=tokens_per_turn)}",
            }
            history.append(user_msg)
            if t < turns - 1:
                assistant_msg = {
                    "role": "assistant",
                    "content": f"Response to turn {t}.",
                }
                history.append(assistant_msg)
            turn_snapshots.append(list(history))
        conversations.append(turn_snapshots)
    return conversations


def generate_rag_context(
    context_tokens: int, num_queries: int, query_tokens: int
) -> list[tuple[dict, dict]]:
    context_msg = {
        "role": "system",
        "content": (
            f"Use the following context to answer the user's question.\n\n"
            f"Context: {_filler_text(seed=3000, approx_tokens=context_tokens)}"
        ),
    }
    requests = []
    for i in range(num_queries):
        query_msg = {
            "role": "user",
            "content": f"Query {i}: {_filler_text(seed=4000 + i, approx_tokens=query_tokens)}",
        }
        requests.append((context_msg, query_msg))
    return requests


def generate_cache_pressure(
    prefix_tokens: int, unique_prefixes: list[int], requests_per_prefix: int
) -> list[list[tuple[dict, dict]]]:
    batches = []
    for n_prefixes in unique_prefixes:
        batch = []
        for p in range(n_prefixes):
            system_msg = {
                "role": "system",
                "content": f"Prefix {p}: {_filler_text(seed=5000 + p, approx_tokens=prefix_tokens)}",
            }
            for r in range(requests_per_prefix):
                user_msg = {
                    "role": "user",
                    "content": f"Request {r} for prefix {p}.",
                }
                batch.append((system_msg, user_msg))
        batches.append(batch)
    return batches
