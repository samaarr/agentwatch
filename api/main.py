"""
AgentWatch API

Three endpoints:
  POST /run              — start agent on a task, returns run_id
  GET  /traces/{run_id}  — full step-by-step trace with cycle annotations
  GET  /alerts/{run_id}  — just the detected cycles, clean JSON

Run with: uvicorn api.main:app --reload --port 8000
"""

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent.base_agent import run_agent
from agent.tools import ToolBehaviour
from detector import analyse_trajectory
from detector.models import DetectionMethod

app = FastAPI(
    title="AgentWatch",
    description="Runtime cycle detector for LLM agents — implements George et al. ICPE 2026",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory run store (sufficient for demo; swap for Redis in production)
_runs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    task: str = "Find the population of Dublin and calculate its percentage of Ireland's total population."
    scenario: str = "productive"  # "productive" | "error_cycle" | "silent_cycle"
    phi: float = 0.92
    k: float = 2.0
    m: float = 2.0


class RunResponse(BaseModel):
    run_id: str
    trace_id: str
    label: str
    is_bad_cycle: bool
    span_count: int
    alert_count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest):
    """
    Execute the demo agent on a task and analyse the resulting trajectory.
    Returns run_id for subsequent trace/alert lookups.
    """
    # Map scenario name to tool behaviours
    scenario_map = {
        "productive": (ToolBehaviour.NORMAL, ToolBehaviour.NORMAL),
        "error_cycle": (ToolBehaviour.EMPTY, ToolBehaviour.NORMAL),
        "silent_cycle": (ToolBehaviour.NORMAL, ToolBehaviour.WRONG_TYPE),
    }
    if req.scenario not in scenario_map:
        raise HTTPException(status_code=400, detail=f"Unknown scenario '{req.scenario}'. Choose: {list(scenario_map.keys())}")

    search_b, calc_b = scenario_map[req.scenario]

    # Run agent and collect spans
    spans, final_answer = run_agent(
        task=req.task,
        search_behaviour=search_b,
        calc_behaviour=calc_b,
        max_steps=12,
    )

    # Analyse trajectory
    result = analyse_trajectory(
        spans,
        method=DetectionMethod.HYBRID,
        phi=req.phi,
        k=req.k,
        m=req.m,
    )

    run_id = str(uuid.uuid4())[:8]

    # Store everything for retrieval
    _runs[run_id] = {
        "run_id": run_id,
        "trace_id": result.trace_id,
        "task": req.task,
        "scenario": req.scenario,
        "final_answer": final_answer,
        "spans": [
            {
                "span_id": s.span_id,
                "op": s.op,
                "input": s.input[:200],
                "output": s.output[:400],
                "trace_id": s.trace_id,
                "parent_span_id": s.parent_span_id,
                "timestamp": s.timestamp,
            }
            for s in spans
        ],
        "result": result.to_dict(),
    }

    return RunResponse(
        run_id=run_id,
        trace_id=result.trace_id,
        label=result.label.value,
        is_bad_cycle=result.is_bad_cycle,
        span_count=result.span_count,
        alert_count=len(result.alerts),
    )


@app.get("/traces/{run_id}")
async def get_trace(run_id: str):
    """Full step-by-step trace with cycle annotations."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")
    run = _runs[run_id]
    return {
        "run_id": run_id,
        "trace_id": run["trace_id"],
        "task": run["task"],
        "scenario": run["scenario"],
        "final_answer": run["final_answer"],
        "spans": run["spans"],
        "analysis": run["result"],
    }


@app.get("/alerts/{run_id}")
async def get_alerts(run_id: str):
    """Just the detected cycle alerts — clean JSON."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")
    run = _runs[run_id]
    result = run["result"]
    return {
        "run_id": run_id,
        "is_bad_cycle": result["is_bad_cycle"],
        "label": result["label"],
        "alerts": result["alerts"],
    }


@app.get("/runs")
async def list_runs():
    """List all runs — useful for dashboard."""
    return [
        {
            "run_id": r["run_id"],
            "scenario": r["scenario"],
            "label": r["result"]["label"],
            "is_bad_cycle": r["result"]["is_bad_cycle"],
            "span_count": r["result"]["span_count"],
            "alert_count": len(r["result"]["alerts"]),
        }
        for r in _runs.values()
    ]


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the AgentWatch dashboard."""
    with open("dashboard/index.html") as f:
        return f.read()
