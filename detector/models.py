"""
Data models faithful to the formal problem definition in:
  George et al., "Unsupervised Cycle Detection in Agentic Applications", ICPE 2026
  arXiv:2511.10650

A trajectory T = {s1, s2, ..., sn} is a collection of spans.
Each span s_i = <trace_id, span_id, parent_span_id, op, input, output>
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class CycleType(str, Enum):
    """
    Failure taxonomy from Pathak et al. (ICPE 2026), arXiv:2511.04032
    Trajectories are one of four types — only Error Cycle and Silent Cycle
    are 'bad cycles' in the sense of George et al.
    """
    PRODUCTIVE = "productive"          # healthy, makes progress
    REDUNDANT_STEP = "redundant_step"  # unnecessary steps but correct output
    ERROR_CYCLE = "error_cycle"        # explicit structural loop / repeated failure
    SILENT_CYCLE = "silent_cycle"      # semantically redundant — same output, different call


class DetectionMethod(str, Enum):
    CDDAG = "cddag"       # Cycle Detection via DAG edge weights
    CDCS = "cdcs"         # Cycle Detection via Call Stack subsequences
    CDSA = "cdsa"         # Cycle Detection via Semantic Analysis
    HYBRID = "hybrid"     # CDCS first, CDSA confirmation (paper's recommended approach)


@dataclass
class Span:
    """
    Formal span definition from Section 2 of George et al.
    s_i = <trace_id, span_id, parent_span_id, op, input, output>
    """
    span_id: str
    op: str                          # operation name, e.g. "web_search", "calculator"
    input: str
    output: str
    trace_id: str = "default"
    parent_span_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def __hash__(self):
        return hash(self.span_id)


@dataclass
class CycleAlert:
    """Emitted when a bad cycle is detected in a trajectory."""
    cycle_type: CycleType
    detection_method: DetectionMethod
    confidence: float                # 0.0 - 1.0
    involved_span_ids: list[str]
    explanation: str
    recommended_action: str

    def to_dict(self) -> dict:
        return {
            "cycle_type": self.cycle_type.value,
            "detection_method": self.detection_method.value,
            "confidence": round(self.confidence, 3),
            "involved_span_ids": self.involved_span_ids,
            "explanation": self.explanation,
            "recommended_action": self.recommended_action,
        }


@dataclass
class TrajectoryResult:
    """Full result for a single trajectory analysis."""
    trace_id: str
    label: CycleType              # final classification
    is_bad_cycle: bool            # f(T) in paper: 1 = bad, 0 = healthy
    alerts: list[CycleAlert]
    span_count: int
    method_used: DetectionMethod

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "label": self.label.value,
            "is_bad_cycle": self.is_bad_cycle,
            "span_count": self.span_count,
            "method_used": self.method_used.value,
            "alerts": [a.to_dict() for a in self.alerts],
        }
