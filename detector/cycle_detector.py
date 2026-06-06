"""
Cycle detector implementing the three methods from George et al. (ICPE 2026):

  - CDDAG: Cycle Detection via DAG edge weight statistics
  - CDCS:  Cycle Detection via Call Stack subsequence frequency
  - CDSA:  Cycle Detection via Semantic (cosine) similarity of sibling spans
  - Hybrid: CDCS first, CDSA confirmation — the paper's recommended approach

Paper benchmark (on 1575 stock market trajectories):
  CDDAG alone:  F1 = 0.08
  CDSA alone:   F1 = 0.28
  Hybrid:       F1 = 0.72  (precision 0.62, recall 0.86)

Tunable parameters (paper notation preserved):
  m  — CDDAG threshold multiplier  (default 2.0)
  k  — CDCS threshold multiplier   (default 2.0)
  phi — CDSA cosine similarity threshold (default 0.92)
"""

import math
import statistics
from collections import defaultdict
from typing import Optional

from .models import (
    Span, CycleType, DetectionMethod, CycleAlert, TrajectoryResult
)
from .pattern_store import PatternStore


# ---------------------------------------------------------------------------
# Optional: sentence-transformers for CDSA (offline, runs on M2)
# Falls back to simple Jaccard similarity if not installed.
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    SEMANTIC_BACKEND = "sentence_transformers"
except ImportError:
    _MODEL = None
    SEMANTIC_BACKEND = "jaccard_fallback"


def _cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two output strings.
    Uses sentence-transformers if available (recommended),
    falls back to Jaccard overlap on word sets.
    """
    if _MODEL is not None:
        emb_a = _MODEL.encode(text_a, convert_to_tensor=True)
        emb_b = _MODEL.encode(text_b, convert_to_tensor=True)
        return float(st_util.cos_sim(emb_a, emb_b)[0][0])
    else:
        # Jaccard fallback — less accurate but zero dependencies
        set_a = set(text_a.lower().split())
        set_b = set(text_b.lower().split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# CDDAG — Cycle Detection via DAG edge weights
# ---------------------------------------------------------------------------

def detect_cddag(store: PatternStore, m: float = 2.0) -> Optional[CycleAlert]:
    """
    Section 2.1, George et al.:
    Flag edge e as cyclic if w(e) > mu + m * sigma.

    Note: paper shows F1=0.08 for this alone. Included for completeness
    and as first stage in hybrid pipeline.
    """
    weights = list(store.get_edge_weights().values())
    if len(weights) < 2:
        return None

    mu = statistics.mean(weights)
    sigma = statistics.pstdev(weights)
    if sigma == 0:
        return None

    threshold = mu + m * sigma
    cyclic_edges = [
        (parent, child)
        for (parent, child), w in store.get_edge_weights().items()
        if w > threshold
    ]

    if not cyclic_edges:
        return None

    involved = list({sid for edge in cyclic_edges for sid in edge})
    max_w = max(store.get_edge_weights()[(p, c)] for p, c in cyclic_edges)
    confidence = min(1.0, (max_w - threshold) / (threshold + 1e-9))

    return CycleAlert(
        cycle_type=CycleType.ERROR_CYCLE,
        detection_method=DetectionMethod.CDDAG,
        confidence=confidence,
        involved_span_ids=involved,
        explanation=(
            f"DAG edge weight {max_w:.1f} exceeds threshold {threshold:.2f} "
            f"(mu={mu:.2f}, sigma={sigma:.2f}, m={m}). "
            f"Agent repeatedly traversed the same parent→child path."
        ),
        recommended_action="INTERRUPT: Break the loop and re-prompt with explicit termination condition.",
    )


# ---------------------------------------------------------------------------
# CDCS — Cycle Detection via Call Stack subsequence frequency
# ---------------------------------------------------------------------------

def _get_subsequences(sequence: list[str], min_len: int = 2) -> dict[tuple, int]:
    """
    Sliding window over op sequence to count all contiguous subsequences.
    Uses tuple of op names as key (paper uses op field for structural identity).
    """
    freq: dict[tuple, int] = defaultdict(int)
    n = len(sequence)
    for length in range(min_len, n + 1):
        for start in range(n - length + 1):
            subseq = tuple(sequence[start:start + length])
            freq[subseq] += 1
    return dict(freq)


def detect_cdcs(store: PatternStore, k: float = 2.0) -> Optional[CycleAlert]:
    """
    Section 2.1, George et al.:
    Flag subsequence S as cyclic if w(S) > mu + k * sigma.
    """
    op_seq = store.get_op_sequence()
    if len(op_seq) < 4:
        return None

    freq = _get_subsequences(op_seq)
    if len(freq) < 2:
        return None

    counts = list(freq.values())
    mu = statistics.mean(counts)
    sigma = statistics.pstdev(counts)
    if sigma == 0:
        return None

    threshold = mu + k * sigma
    cyclic_subseqs = {s: w for s, w in freq.items() if w > threshold}

    if not cyclic_subseqs:
        return None

    worst = max(cyclic_subseqs, key=cyclic_subseqs.get)
    worst_count = cyclic_subseqs[worst]
    confidence = min(1.0, (worst_count - threshold) / (threshold + 1e-9))

    # Find span_ids involved in the repeating subsequence
    call_stack = store.get_call_stack()
    ops = [s.op for s in call_stack]
    involved_ids = []
    pattern = list(worst)
    for i in range(len(ops) - len(pattern) + 1):
        if ops[i:i + len(pattern)] == pattern:
            involved_ids.extend([call_stack[i + j].span_id for j in range(len(pattern))])

    return CycleAlert(
        cycle_type=CycleType.ERROR_CYCLE,
        detection_method=DetectionMethod.CDCS,
        confidence=confidence,
        involved_span_ids=list(set(involved_ids)),
        explanation=(
            f"Call stack pattern {list(worst)} repeated {worst_count}x "
            f"(threshold {threshold:.2f}). Agent is looping over the same tool sequence."
        ),
        recommended_action="INTERRUPT: Detected structural loop in tool call sequence. Inject stop condition.",
    )


# ---------------------------------------------------------------------------
# CDSA — Cycle Detection via Semantic Similarity of sibling spans
# ---------------------------------------------------------------------------

def detect_cdsa(store: PatternStore, phi: float = 0.92) -> Optional[CycleAlert]:
    """
    Section 2.1, George et al.:
    Check cosine similarity between sibling node outputs.
    Flag if cos(v_i, v_j) > phi for any sibling pair.

    Paper restricts to sibling nodes (same parent) to reduce O(n^2) → O(log(n)^2).
    """
    pairs = store.get_all_sibling_pairs()
    if not pairs:
        return None

    best_sim = 0.0
    best_pair: Optional[tuple[Span, Span]] = None

    for span_a, span_b in pairs:
        if not span_a.output.strip() or not span_b.output.strip():
            continue
        sim = _cosine_similarity(span_a.output, span_b.output)
        if sim > best_sim:
            best_sim = sim
            best_pair = (span_a, span_b)

    if best_sim <= phi or best_pair is None:
        return None

    return CycleAlert(
        cycle_type=CycleType.SILENT_CYCLE,
        detection_method=DetectionMethod.CDSA,
        confidence=min(1.0, best_sim),
        involved_span_ids=[best_pair[0].span_id, best_pair[1].span_id],
        explanation=(
            f"Sibling spans '{best_pair[0].op}' and '{best_pair[1].op}' "
            f"have output similarity {best_sim:.3f} > phi={phi}. "
            f"Agent is regenerating semantically identical content."
        ),
        recommended_action="INTERRUPT: Silent cycle — agent is producing redundant outputs. Check if earlier result was consumed.",
    )


# ---------------------------------------------------------------------------
# Hybrid detector — paper's recommended approach
# ---------------------------------------------------------------------------

def detect_hybrid(
    store: PatternStore,
    m: float = 2.0,
    k: float = 2.0,
    phi: float = 0.92,
) -> TrajectoryResult:
    """
    Hybrid approach from Section 2.1:
    1. Run CDCS (call stack structural analysis)
    2. If structural cycle found, confirm with CDSA
    3. Also run CDDAG independently

    Returns a TrajectoryResult with full classification.
    """
    alerts: list[CycleAlert] = []

    # Stage 1: structural detection via call stack
    cdcs_alert = detect_cdcs(store, k=k)

    # Stage 2: semantic confirmation (only if structural cycle suspected,
    # or run independently to catch silent cycles structural methods miss)
    cdsa_alert = detect_cdsa(store, phi=phi)

    # Stage 3: DAG structural check (lower precision, catches explicit repeats)
    cddag_alert = detect_cddag(store, m=m)

    # Collect all alerts
    if cdcs_alert:
        alerts.append(cdcs_alert)
    if cdsa_alert:
        alerts.append(cdsa_alert)
    if cddag_alert:
        # Only add CDDAG if it found something the others didn't
        if not cdcs_alert:
            alerts.append(cddag_alert)

    # Determine final label
    if not alerts:
        label = CycleType.PRODUCTIVE
        is_bad = False
    elif any(a.cycle_type == CycleType.SILENT_CYCLE for a in alerts):
        label = CycleType.SILENT_CYCLE
        is_bad = True
    else:
        label = CycleType.ERROR_CYCLE
        is_bad = True

    # Determine trace_id from spans
    call_stack = store.get_call_stack()
    trace_id = call_stack[0].trace_id if call_stack else "unknown"

    return TrajectoryResult(
        trace_id=trace_id,
        label=label,
        is_bad_cycle=is_bad,
        alerts=alerts,
        span_count=store.span_count(),
        method_used=DetectionMethod.HYBRID,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyse_trajectory(
    spans: list[Span],
    method: DetectionMethod = DetectionMethod.HYBRID,
    m: float = 2.0,
    k: float = 2.0,
    phi: float = 0.92,
) -> TrajectoryResult:
    """
    Main entry point. Takes a completed trajectory (list of Span),
    builds internal representations, runs detection.

    Args:
        spans:  All spans in the trajectory
        method: Which detection method to use (default: HYBRID)
        m:      CDDAG threshold multiplier
        k:      CDCS threshold multiplier
        phi:    CDSA cosine similarity threshold

    Returns:
        TrajectoryResult with label, is_bad_cycle flag, and alerts
    """
    store = PatternStore()
    for span in spans:
        store.add_span(span)

    if method == DetectionMethod.HYBRID:
        return detect_hybrid(store, m=m, k=k, phi=phi)

    # Single-method runs (for benchmarking individual methods)
    trace_id = spans[0].trace_id if spans else "unknown"
    alert = None

    if method == DetectionMethod.CDDAG:
        alert = detect_cddag(store, m=m)
    elif method == DetectionMethod.CDCS:
        alert = detect_cdcs(store, k=k)
    elif method == DetectionMethod.CDSA:
        alert = detect_cdsa(store, phi=phi)

    alerts = [alert] if alert else []
    is_bad = bool(alerts)
    label = alerts[0].cycle_type if alerts else CycleType.PRODUCTIVE

    return TrajectoryResult(
        trace_id=trace_id,
        label=label,
        is_bad_cycle=is_bad,
        alerts=alerts,
        span_count=store.span_count(),
        method_used=method,
    )
