"""
The three detection methods from George et al. (ICPE 2026), plus the hybrid
pipeline that combines them.

Quick reference on what each method is actually good at:
  CDDAG  — catches explicit repeated edges in the DAG. Low precision on its own
           (paper F1: 0.08) because a single busy parent looks like a cycle.
  CDCS   — sliding window over the op sequence. Good at structural loops like
           [web_search -> llm_reasoning] repeating four times.
  CDSA   — cosine similarity between sibling span outputs. The only method that
           catches silent cycles, where the agent tries different things but
           keeps getting the same result. (paper F1 alone: 0.28)
  Hybrid — run CDCS and CDSA in parallel, use CDDAG as a tiebreaker.
           Paper F1: 0.72 (precision 0.62, recall 0.86).

Parameters follow the paper's notation:
  m   — CDDAG threshold multiplier (default 2.0)
  k   — CDCS threshold multiplier  (default 2.0)
  phi — CDSA cosine similarity cutoff (default 0.92)
"""

import statistics
from collections import defaultdict
from typing import Optional

from .models import (
    Span, CycleType, DetectionMethod, CycleAlert, TrajectoryResult
)
from .pattern_store import PatternStore


# sentence-transformers runs offline on M2 via MPS — preferred for CDSA.
# If it's not installed, Jaccard overlap is a reasonable fallback for testing.
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    SEMANTIC_BACKEND = "sentence_transformers"
except ImportError:
    _MODEL = None
    SEMANTIC_BACKEND = "jaccard_fallback"


def _cosine_similarity(text_a: str, text_b: str) -> float:
    if _MODEL is not None:
        emb_a = _MODEL.encode(text_a, convert_to_tensor=True)
        emb_b = _MODEL.encode(text_b, convert_to_tensor=True)
        return float(st_util.cos_sim(emb_a, emb_b)[0][0])
    else:
        set_a = set(text_a.lower().split())
        set_b = set(text_b.lower().split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)


def detect_cddag(store: PatternStore, m: float = 2.0) -> Optional[CycleAlert]:
    """
    Flag edge e as cyclic if w(e) > mu + m * sigma (Section 2.1).

    Works well when an agent genuinely hammers the same parent->child path.
    Less useful in shallow single-agent trees where most edges appear once —
    the variance is too low to distinguish a real loop from normal branching.
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
            f"Agent repeatedly traversed the same parent->child path."
        ),
        recommended_action="INTERRUPT: re-prompt with an explicit termination condition.",
    )


def _get_subsequences(sequence: list[str], min_len: int = 2) -> dict[tuple, int]:
    """
    Count all contiguous subsequences of length >= min_len.
    Keys are tuples of op names — structure only, no content.
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
    Flag subsequence S as cyclic if w(S) > mu + k * sigma (Section 2.1).

    This is the workhorse for error cycles. If an agent is stuck rephrasing
    a query and retrying, you'll see ['llm_reasoning', 'web_search'] show up
    three or four times with a frequency well above the mean.
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
        recommended_action="INTERRUPT: structural loop detected. Inject a stop condition or clear the agent's context.",
    )


def detect_cdsa(store: PatternStore, phi: float = 0.92) -> Optional[CycleAlert]:
    """
    Flag sibling spans as a silent cycle if cos(vi, vj) > phi (Section 2.1).

    Silent cycles are harder to catch than error cycles because the op sequence
    looks fine — the agent is trying different things. But the outputs are
    nearly identical, which means it's not making progress. Restricting to
    siblings (same parent in the DAG) keeps this from being O(n^2).
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
        recommended_action="INTERRUPT: silent cycle — check whether an earlier result was actually consumed.",
    )


def detect_hybrid(
    store: PatternStore,
    m: float = 2.0,
    k: float = 2.0,
    phi: float = 0.92,
) -> TrajectoryResult:
    """
    Run all three methods and merge results.

    CDCS and CDSA run independently — CDDAG is only added when CDCS doesn't
    fire, to avoid double-counting the same loop. Silent cycle takes priority
    in the final label because it's harder to detect and more expensive when
    missed (token cost accumulates invisibly).
    """
    alerts: list[CycleAlert] = []

    cdcs_alert = detect_cdcs(store, k=k)
    cdsa_alert = detect_cdsa(store, phi=phi)
    cddag_alert = detect_cddag(store, m=m)

    if cdcs_alert:
        alerts.append(cdcs_alert)
    if cdsa_alert:
        alerts.append(cdsa_alert)
    if cddag_alert and not cdcs_alert:
        alerts.append(cddag_alert)

    if not alerts:
        label = CycleType.PRODUCTIVE
        is_bad = False
    elif any(a.cycle_type == CycleType.SILENT_CYCLE for a in alerts):
        label = CycleType.SILENT_CYCLE
        is_bad = True
    else:
        label = CycleType.ERROR_CYCLE
        is_bad = True

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


def analyse_trajectory(
    spans: list[Span],
    method: DetectionMethod = DetectionMethod.HYBRID,
    m: float = 2.0,
    k: float = 2.0,
    phi: float = 0.92,
) -> TrajectoryResult:
    """
    Main entry point. Pass in a completed list of spans, get back a
    TrajectoryResult with a label and any alerts.

    Use method=HYBRID in production. The single-method options (CDDAG, CDCS,
    CDSA) are useful for benchmarking individual methods against each other,
    which is how the paper's Table 2 numbers were generated.
    """
    store = PatternStore()
    for span in spans:
        store.add_span(span)

    if method == DetectionMethod.HYBRID:
        return detect_hybrid(store, m=m, k=k, phi=phi)

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