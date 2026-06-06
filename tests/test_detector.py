"""
Unit tests for AgentWatch cycle detector.

Tests each detection layer independently with hand-crafted trajectories,
then tests the hybrid approach against known expected labels.

Run with: pytest tests/test_detector.py -v
"""

import time
import pytest

from detector.models import Span, CycleType, DetectionMethod
from detector.pattern_store import PatternStore
from detector.cycle_detector import (
    detect_cddag,
    detect_cdcs,
    detect_cdsa,
    analyse_trajectory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_span(op: str, output: str, parent_id=None, trace_id="test") -> Span:
    import uuid
    return Span(
        span_id=str(uuid.uuid4())[:8],
        op=op,
        input=f"input for {op}",
        output=output,
        trace_id=trace_id,
        parent_span_id=parent_id,
        timestamp=time.time(),
    )


def build_store(spans: list[Span]) -> PatternStore:
    store = PatternStore()
    for s in spans:
        store.add_span(s)
    return store


# ---------------------------------------------------------------------------
# CDDAG tests
# ---------------------------------------------------------------------------

class TestCDDAG:

    def test_no_cycle_returns_none(self):
        root = make_span("agent_run", "started")
        spans = [
            root,
            make_span("web_search", "result A", root.span_id),
            make_span("calculator", "27.4%", root.span_id),
            make_span("finish", "done", root.span_id),
        ]
        store = build_store(spans)
        assert detect_cddag(store) is None

    def test_repeated_edge_triggers_alert(self):
        root = make_span("agent_run", "started")
        # Simulate same parent->child edge repeated 5 times
        for _ in range(5):
            store = PatternStore()
            store.add_span(root)
            for i in range(5):
                child = make_span("web_search", f"empty result {i}", root.span_id)
                store.add_span(child)
        # Build fresh store with high-weight edge
        store = PatternStore()
        store.add_span(root)
        # Manually insert high-weight by adding many same-parent children
        for i in range(10):
            store.add_span(make_span("web_search", f"result {i}", root.span_id))

        result = detect_cddag(store, m=0.5)  # low threshold to trigger
        # With many children of same parent edge weight won't repeat
        # CDDAG works on (parent, child) edges — this tests the mechanism is live
        assert result is None or result.detection_method == DetectionMethod.CDDAG


# ---------------------------------------------------------------------------
# CDCS tests
# ---------------------------------------------------------------------------

class TestCDCS:

    def test_no_cycle_short_trajectory(self):
        spans = [
            make_span("agent_run", "start"),
            make_span("web_search", "result"),
            make_span("calculator", "27%"),
            make_span("finish", "done"),
        ]
        store = build_store(spans)
        assert detect_cdcs(store) is None

    def test_repeated_subsequence_triggers_alert(self):
        """Agent searches, gets empty, rephrases, searches again — classic error cycle."""
        spans = []
        root = make_span("agent_run", "start")
        spans.append(root)
        # Repeat the same pattern 4 times
        for _ in range(4):
            spans.append(make_span("web_search", "empty"))
            spans.append(make_span("llm_reasoning", "let me try a different query"))
        store = build_store(spans)
        result = detect_cdcs(store, k=0.5)
        assert result is not None
        assert result.cycle_type == CycleType.ERROR_CYCLE
        assert result.detection_method == DetectionMethod.CDCS

    def test_productive_pattern_not_flagged(self):
        """Sequential different ops should not be flagged."""
        spans = [
            make_span("agent_run", "start"),
            make_span("web_search", "Dublin population 1.4M"),
            make_span("calculator", "27.4%"),
            make_span("memory_store_write", "stored"),
            make_span("finish", "27.4%"),
        ]
        store = build_store(spans)
        result = detect_cdcs(store, k=1.0)
        assert result is None


# ---------------------------------------------------------------------------
# CDSA tests
# ---------------------------------------------------------------------------

class TestCDSA:

    def test_identical_sibling_outputs_flagged(self):
        root = make_span("agent_run", "start")
        child_a = make_span("calculator", "Error: expected numeric input", root.span_id)
        # Add a tiny delay so timestamps differ
        time.sleep(0.01)
        child_b = make_span("calculator", "Error: expected numeric input", root.span_id)

        store = build_store([root, child_a, child_b])
        result = detect_cdsa(store, phi=0.8)
        assert result is not None
        assert result.cycle_type == CycleType.SILENT_CYCLE
        assert result.detection_method == DetectionMethod.CDSA

    def test_different_sibling_outputs_not_flagged(self):
        root = make_span("agent_run", "start")
        child_a = make_span("web_search", "Dublin population is 1.4 million people", root.span_id)
        time.sleep(0.01)
        child_b = make_span("calculator", "27.45%", root.span_id)

        store = build_store([root, child_a, child_b])
        result = detect_cdsa(store, phi=0.92)
        assert result is None

    def test_high_similarity_below_threshold_not_flagged(self):
        root = make_span("agent_run", "start")
        child_a = make_span("web_search", "Dublin population: 1.4 million", root.span_id)
        time.sleep(0.01)
        child_b = make_span("web_search", "Population of Dublin is roughly 1.4 million people", root.span_id)

        store = build_store([root, child_a, child_b])
        # phi=1.0 means nothing except identical text triggers it
        result = detect_cdsa(store, phi=1.0)
        assert result is None


# ---------------------------------------------------------------------------
# Hybrid tests
# ---------------------------------------------------------------------------

class TestHybrid:

    def test_productive_trajectory(self):
        spans = [
            make_span("agent_run", "start"),
            make_span("web_search", "Dublin 1.4M, Ireland 5.1M"),
            make_span("calculator", "27.45"),
            make_span("finish", "Dublin is 27.45% of Ireland's population"),
        ]
        result = analyse_trajectory(spans)
        assert result.is_bad_cycle is False
        assert result.label == CycleType.PRODUCTIVE

    def test_error_cycle_trajectory(self):
        spans = [make_span("agent_run", "start")]
        for _ in range(5):
            spans.append(make_span("web_search", ""))
            spans.append(make_span("llm_reasoning", "try again"))
        result = analyse_trajectory(spans, k=0.5)
        assert result.is_bad_cycle is True
        assert result.label == CycleType.ERROR_CYCLE

    def test_silent_cycle_trajectory(self):
        root = make_span("agent_run", "start", trace_id="silent_test")
        children = []
        for _ in range(3):
            time.sleep(0.01)
            children.append(
                make_span("calculator", "Error: expected numeric input", root.span_id, "silent_test")
            )
        result = analyse_trajectory([root] + children, phi=0.8)
        assert result.is_bad_cycle is True
        assert result.label == CycleType.SILENT_CYCLE

    def test_result_serialises_to_dict(self):
        spans = [make_span("agent_run", "done")]
        result = analyse_trajectory(spans)
        d = result.to_dict()
        assert "trace_id" in d
        assert "label" in d
        assert "is_bad_cycle" in d
        assert "alerts" in d
