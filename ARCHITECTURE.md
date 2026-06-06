# AgentWatch — Architecture

**Reference implementation of George et al., "Unsupervised Cycle Detection in Agentic Applications", ICPE 2026**
arXiv:2511.10650

---

## Scope

AgentWatch implements the two-stage hybrid cycle detection methodology from George et al. as a standalone Python library with a FastAPI observability layer.

**In scope:**
- CDDAG: Cycle Detection via DAG edge weight statistics
- CDCS: Cycle Detection via Call Stack subsequence frequency
- CDSA: Cycle Detection via cosine similarity of sibling span outputs
- Hybrid pipeline (CDCS → CDSA confirmation) — the paper's recommended approach
- Failure taxonomy from Pathak et al. ICPE 2026 (arXiv:2511.04032): Productive / Redundant Step / Error Cycle / Silent Cycle

**Out of scope (deliberately):**
- Multi-agent orchestration (paper uses LangGraph; we use raw API for simplicity)
- Pre-execution guardrail layer (Huang et al. ICLR 2026, arXiv:2510.09781) — future work
- Fix recommendation engine (AgentFixer, Mulian et al. ICSE 2026) — future work
- Full replication of paper's 1575-trajectory stock market benchmark

---

## Formal Problem Definition

*Preserved from Section 2, George et al.*

An agent execution trajectory **T** = {s₁, s₂, ..., sₙ} is a collection of spans, where each span sᵢ = ⟨trace_id, span_id, parent_span_id, op, input, output⟩.

The detection task is a binary classification:
- f(T) = 1 → trajectory contains a bad cycle
- f(T) = 0 → trajectory is non-cyclical or productive

---

## Detection Layers

### Layer 1 — CDDAG (DAG edge weight analysis)
Flag edge e as cyclic if `w(e) > μ + m·σ` where μ, σ are mean and std of all edge weights.
Paper F1 alone: **0.08** — low precision, used as supporting signal only.

### Layer 2 — CDCS (Call stack subsequence frequency)
Build sliding-window frequency map of all contiguous op subsequences.
Flag if any subsequence frequency exceeds `μ + k·σ`.
Detects **Error Cycles** — explicit repeated tool sequences.

### Layer 3 — CDSA (Semantic similarity of siblings)
Compute cosine similarity between sibling span outputs (same parent in DAG).
Flag if `cos(vᵢ, vⱼ) > φ` for any sibling pair.
Restricts to siblings (not all pairs) to reduce O(n²) → O(log(n)²).
Detects **Silent Cycles** — semantically redundant outputs.
Paper F1 alone: **0.28**.

### Hybrid (recommended)
1. Run CDCS structural analysis
2. Run CDSA independently (catches silent cycles structural methods miss)
3. Run CDDAG as supplementary signal
Paper F1 hybrid: **0.72** (precision 0.62, recall 0.86)

---

## Design Decisions

**ADR-001: Raw Anthropic API over LangGraph**
Paper uses LangGraph. We use raw Anthropic tool-use API to keep the implementation
framework-agnostic and dependency-light for M2 Mac development.
Tradeoff: span tree is shallower (less multi-agent nesting). Detection logic unchanged.

**ADR-002: sentence-transformers over OpenAI embeddings**
Paper uses OpenAI embeddings for CDSA. We use `all-MiniLM-L6-v2` via
sentence-transformers: fully offline, runs on M2 MPS, zero API cost.
Tradeoff: slightly lower embedding quality. Acceptable for demo purposes.
Jaccard fallback provided if sentence-transformers not installed.

**ADR-003: In-memory run store**
Production would use Redis or PostgreSQL. For demo clarity, runs stored in-process dict.
No state persistence between API restarts. Explicitly documented limitation.

**ADR-004: Thresholds as configurable parameters**
Paper tunes m, k, φ empirically on their dataset. We expose all three as API parameters
so users can experiment. Defaults match paper recommendations.

---

## Component Map

```
agentwatch/
├── detector/
│   ├── models.py          Span, CycleType, CycleAlert, TrajectoryResult
│   ├── pattern_store.py   Call stack + DAG representations
│   └── cycle_detector.py  CDDAG, CDCS, CDSA, Hybrid — core algorithms
├── agent/
│   ├── base_agent.py      ReAct-style demo agent, emits spans
│   └── tools.py           Mock tools with injectable failure modes
├── api/
│   └── main.py            FastAPI: /run, /traces, /alerts, /runs
├── dashboard/
│   └── index.html         Single-file dashboard, Chart.js
└── tests/
    ├── test_detector.py   Unit tests per detection layer
    └── scenarios/         Three benchmark scenarios
```

---

## Benchmark Scenarios

| Scenario | Tool Config | Expected Label | Detection Method |
|----------|------------|----------------|------------------|
| Scenario 1 | web_search → empty | ERROR_CYCLE | CDCS |
| Scenario 2 | calculator → wrong type | SILENT_CYCLE | CDSA |
| Scenario 3 | all tools normal | PRODUCTIVE | — |

---

## Limitations

1. **Shallow span tree**: Single-agent ReAct produces a flatter DAG than the paper's hierarchical LangGraph setup. CDDAG has less signal.
2. **Small benchmark**: Three handcrafted scenarios vs. paper's 1575 trajectories. F1 comparison not directly meaningful.
3. **No streaming detection**: Current design analyses completed trajectories. Real-time detection (interrupt mid-run) is future work.
4. **Jaccard fallback**: If sentence-transformers unavailable, CDSA uses word-overlap similarity — lower accuracy.

---

## References

1. George, F. et al. "Unsupervised Cycle Detection in Agentic Applications." ICPE 2026. arXiv:2511.10650
2. Pathak, D. et al. "Detecting Silent Failures in Multi-Agentic AI Trajectories." ICPE 2026. arXiv:2511.04032
3. Mulian, H. et al. "AgentFixer: From Failure Detection to Fix Recommendations in Agentic Systems." ICSE 2026.
4. Huang, Y. et al. "Building a Foundational Guardrail for General Agentic Systems via Synthetic Data." ICLR 2026. arXiv:2510.09781
