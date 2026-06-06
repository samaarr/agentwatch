"""
Maintains the two trajectory representations the paper needs:

  - Call stack C_T: spans ordered by creation time. Used by CDCS to find
    repeating op subsequences.

  - DAG G_T: parent-child edges with weights. Used by CDDAG (edge frequency)
    and CDSA (sibling pairs for cosine similarity).

Spans are added one at a time as the agent executes, so both structures
stay up to date without a second pass.
"""

from collections import defaultdict
from typing import Optional
from .models import Span


class PatternStore:

    def __init__(self):
        self._call_stack: list[Span] = []
        self._edge_weights: dict[tuple[str, str], int] = defaultdict(int)
        self._children: dict[str, list[str]] = defaultdict(list)
        self._spans: dict[str, Span] = {}

    def add_span(self, span: Span) -> None:
        self._spans[span.span_id] = span

        self._call_stack.append(span)
        self._call_stack.sort(key=lambda s: s.timestamp)

        if span.parent_span_id is not None:
            edge = (span.parent_span_id, span.span_id)
            self._edge_weights[edge] += 1
            self._children[span.parent_span_id].append(span.span_id)

    def get_call_stack(self) -> list[Span]:
        return list(self._call_stack)

    def get_op_sequence(self) -> list[str]:
        """Operation names in temporal order — the input to CDCS."""
        return [s.op for s in self._call_stack]

    def get_edge_weights(self) -> dict[tuple[str, str], int]:
        return dict(self._edge_weights)

    def get_siblings(self, span_id: str) -> list[Span]:
        """
        Spans sharing the same parent as span_id.
        The paper restricts CDSA to sibling pairs rather than all pairs,
        which cuts comparisons from O(n^2) down to O(log(n)^2).
        """
        span = self._spans.get(span_id)
        if span is None or span.parent_span_id is None:
            return []

        sibling_ids = self._children.get(span.parent_span_id, [])
        return [
            self._spans[sid]
            for sid in sibling_ids
            if sid != span_id and sid in self._spans
        ]

    def get_all_sibling_pairs(self) -> list[tuple[Span, Span]]:
        """All (a, b) pairs where a and b share a parent. Input to CDSA."""
        pairs = []
        seen = set()
        for parent_id, child_ids in self._children.items():
            for i in range(len(child_ids)):
                for j in range(i + 1, len(child_ids)):
                    a, b = child_ids[i], child_ids[j]
                    key = tuple(sorted([a, b]))
                    if key not in seen and a in self._spans and b in self._spans:
                        pairs.append((self._spans[a], self._spans[b]))
                        seen.add(key)
        return pairs

    def get_span(self, span_id: str) -> Optional[Span]:
        return self._spans.get(span_id)

    def span_count(self) -> int:
        return len(self._spans)

    def reset(self) -> None:
        self._call_stack.clear()
        self._edge_weights.clear()
        self._children.clear()
        self._spans.clear()