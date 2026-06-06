"""
Benchmark scenarios for AgentWatch.

Three injectable failure modes corresponding to Pathak et al. (ICPE 2026)
failure taxonomy: Error Cycle, Silent Cycle, Productive.

These form the test suite used to validate the detector.
"""

from dataclasses import dataclass
from ..agent.tools import ToolBehaviour
from ..detector.models import CycleType


@dataclass
class Scenario:
    name: str
    description: str
    task: str
    search_behaviour: ToolBehaviour
    calc_behaviour: ToolBehaviour
    expected_label: CycleType
    paper_reference: str


SCENARIOS = {

    "scenario_1_error_cycle": Scenario(
        name="Scenario 1 — Error Cycle (Search Loop)",
        description=(
            "web_search always returns empty results. "
            "Agent enters an Error Cycle: it rephrases the same query "
            "repeatedly without making progress. "
            "Expected: CDCS detects the repeating 'web_search' subsequence."
        ),
        task="Find the population of Dublin and calculate its percentage of Ireland's total population.",
        search_behaviour=ToolBehaviour.EMPTY,
        calc_behaviour=ToolBehaviour.NORMAL,
        expected_label=CycleType.ERROR_CYCLE,
        paper_reference="George et al. ICPE 2026, Section 3.1 — Error Cycle label",
    ),

    "scenario_2_silent_cycle": Scenario(
        name="Scenario 2 — Silent Cycle (Redundant Output)",
        description=(
            "calculator returns a wrong-type error. "
            "Agent enters a Silent Cycle: it retries calculator with identical "
            "or semantically equivalent expressions, producing the same error output. "
            "Expected: CDSA detects high cosine similarity between sibling calculator spans."
        ),
        task="Calculate what percentage 1.4 million is of 5.1 million.",
        search_behaviour=ToolBehaviour.NORMAL,
        calc_behaviour=ToolBehaviour.WRONG_TYPE,
        expected_label=CycleType.SILENT_CYCLE,
        paper_reference="George et al. ICPE 2026, Section 3.1 — Silent Cycle label",
    ),

    "scenario_3_productive": Scenario(
        name="Scenario 3 — Productive (Healthy Trajectory)",
        description=(
            "All tools work correctly. Agent completes the task efficiently "
            "without looping. "
            "Expected: PRODUCTIVE — no cycle detected."
        ),
        task="Find the population of Dublin and calculate its percentage of Ireland's total population.",
        search_behaviour=ToolBehaviour.NORMAL,
        calc_behaviour=ToolBehaviour.NORMAL,
        expected_label=CycleType.PRODUCTIVE,
        paper_reference="George et al. ICPE 2026, Section 3.1 — Productive label",
    ),
}
