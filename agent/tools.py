"""
Mock tools used by the demo agent. Each tool can be configured to behave
normally or fail in a specific way, which is how we inject the failure
scenarios for the benchmark.

Nothing here makes real network calls — that keeps tests fast and costs zero.
"""

from dataclasses import dataclass
from enum import Enum


class ToolBehaviour(str, Enum):
    NORMAL = "normal"
    EMPTY = "empty"            # web_search returns nothing -> search loop
    WRONG_TYPE = "wrong_type"  # calculator rejects input -> retry loop
    CORRECT_NO_STOP = "correct_no_stop"


@dataclass
class ToolResult:
    content: str
    success: bool
    tool_name: str


class MockWebSearch:
    def __init__(self, behaviour: ToolBehaviour = ToolBehaviour.NORMAL):
        self.behaviour = behaviour
        self.call_count = 0

    def __call__(self, query: str) -> ToolResult:
        self.call_count += 1

        if self.behaviour == ToolBehaviour.EMPTY:
            return ToolResult(content="", success=False, tool_name="web_search")

        return ToolResult(
            content=f"Search results for '{query}': Dublin population is approximately 1.4 million. Ireland total population is 5.1 million.",
            success=True,
            tool_name="web_search",
        )


class MockCalculator:
    def __init__(self, behaviour: ToolBehaviour = ToolBehaviour.NORMAL):
        self.behaviour = behaviour
        self.call_count = 0

    def __call__(self, expression: str) -> ToolResult:
        self.call_count += 1

        if self.behaviour == ToolBehaviour.WRONG_TYPE:
            return ToolResult(
                content="Error: expected numeric input",
                success=False,
                tool_name="calculator",
            )

        try:
            result = eval(expression, {"__builtins__": {}})  # noqa: S307 — demo only
            return ToolResult(content=str(result), success=True, tool_name="calculator")
        except Exception as e:
            return ToolResult(content=f"Calculation error: {e}", success=False, tool_name="calculator")


class MockMemoryStore:
    def __init__(self):
        self._store: dict[str, str] = {}
        self.call_count = 0

    def write(self, key: str, value: str) -> ToolResult:
        self.call_count += 1
        self._store[key] = value
        return ToolResult(content=f"Stored: {key}", success=True, tool_name="memory_store")

    def read(self, key: str) -> ToolResult:
        self.call_count += 1
        value = self._store.get(key, "")
        return ToolResult(
            content=value if value else f"Key '{key}' not found",
            success=bool(value),
            tool_name="memory_store",
        )