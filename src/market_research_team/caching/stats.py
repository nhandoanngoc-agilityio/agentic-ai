"""Process-local cache hit/miss counters, independent of Langfuse.

Used by the eval regression guard to read a synchronous, offline hit rate
without depending on tracing being configured.
"""

from collections import Counter

_counts: Counter[str] = Counter()


def record(layer: str, hit: bool) -> None:
    _counts[f"{layer}.{'hit' if hit else 'miss'}"] += 1


def hit_rate(layer: str | None = None) -> float:
    if layer is None:
        hits = sum(v for k, v in _counts.items() if k.endswith(".hit"))
        total = sum(_counts.values())
    else:
        hits = _counts[f"{layer}.hit"]
        total = hits + _counts[f"{layer}.miss"]
    return hits / total if total else 0.0


def reset() -> None:
    _counts.clear()
