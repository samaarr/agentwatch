"""
Data models for AgentWatch — kept close to the paper's notation so the
code and the maths stay easy to cross-reference.

George et al. define a trajectory as T = {s1, s2, ..., sn} where each
span si = <trace_id, span_id, parent_span_id, op, input, output>.
That's what the Span dataclass encodes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class CycleType(str, Enum):
    # Failure taxonomy from Pathak et al. (ICPE 2026), arXiv:2511.04032.
    # The first two are healthy; the last two are what we're hunting.
    PRODUCTIVE = "productive"          # agent makes progress, finishes cleanly
    REDUNDANT_STEP = "redundant_step"  # extra steps, but correct output
    ERROR_CYCLE = "error_cycle"        # structural loop — same tool sequence repeating
    SILENT_CYCLE = "silent_cycle"      # different calls, same outputs — the sneaky one


class DetectionMethod(str, Enum):
    CDDAG = "cddag"   # DAG edge weights
    CDCS = "cdcs"     # call stack subsequence frequency
    CDSA = "cdsa"     # cosine similarity between sibling spans
    HYBRID = "hybrid" # CDCS + CDSA — what the paper actually recommends


@dataclass
class Span:
    """
    One step in an agent's execution. Maps directly to the paper's formal
    definition: si = <trace_id, span_id, parent_span_id, op, input, output>.
    """
    span_id: str
    op: str          # e.g. "web_search", "calculator", "llm_reasoning"
    input: str
    output: str
    trace_id: str = "default"
    parent_span_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def __hash__(self):
        return hash(self.span_id)


@dataclass
class CycleAlert:
    """Fired when the detector finds a bad cycle in a trajectory."""
    cycle_type: CycleType
    detection_method: DetectionMethod
    confidence: float          # 0.0 - 1.0
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
    """What you get back after running analyse_trajectory()."""
    trace_id: str
    label: CycleType    # f(T) in the paper: bad cycle or not
    is_bad_cycle: bool
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