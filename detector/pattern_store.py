"""
Pattern store: maintains the call stack and DAG representations of a trajectory
as spans arrive, enabling both CDCS and CDDAG analysis.

Based on Section 2.1 of George et al. (ICPE 2026), arXiv:2511.10650
"""

from collections import defaultdict
from typing import Optional
from .models import Span


class PatternStore:
    """
    Maintains two representations of an agent trajectory as it executes:

    1. Call Stack C_T = [s1, s2, ..., sn] ordered by creation time
       Used for CDCS — sliding window subsequence frequency analysis

    2. DAG G_T = (S, E, w) where w(e) = frequency of parent->child edge
       Used for CDDAG — edge weight threshold analysis
    """

    def __init__(self):
        # Call stack: ordered list of spans by timestamp
        self._call_stack: list[Span] = []

        # DAG edge weights: (parent_span_id, child_span_id) -> frequency
        self._edge_weights: dict[tuple[str, str], int] = defaultdict(int)

        # DAG adjacency: span_id -> list of child span_ids
        self._children: dict[str, list[str]] = defaultdict(list)

        # Quick lookup: span_id -> Span
        self._spans: dict[str, Span] = {}

    def add_span(self, span: Span) -> None:
        """Add a span to both representations as it arrives."""
        self._spans[span.span_id] = span

        # Update call stack (temporal order)
        self._call_stack.append(span)
        self._call_stack.sort(key=lambda s: s.timestamp)

        # Update DAG edge weights
        if span.parent_span_id is not None:
            edge = (span.parent_span_id, span.span_id)
            self._edge_weights[edge] += 1
            self._children[span.parent_span_id].append(span.span_id)

    def get_call_stack(self) -> list[Span]:
        return list(self._call_stack)

    def get_op_sequence(self) -> list[str]:
        """Return just the operation names in temporal order — the key signal for CDCS."""
        return [s.op for s in self._call_stack]

    def get_edge_weights(self) -> dict[tuple[str, str], int]:
        return dict(self._edge_weights)

    def get_siblings(self, span_id: str) -> list[Span]:
        """
        Return sibling spans (shares same parent).
        Used in CDSA — paper restricts cosine similarity to sibling nodes only,
        reducing complexity from O(n^2) to O(log(n)^2).
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
        """
        Return all (span_a, span_b) pairs that share a parent.
        These are the pairs subject to cosine similarity in CDSA.
        """
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
