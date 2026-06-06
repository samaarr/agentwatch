"""
A minimal ReAct-style agent that emits spans as it executes.
Spans are fed directly to the PatternStore so the detector
can analyse the trajectory in real time or post-hoc.

The agent's task: "Find the population of Dublin and calculate
its percentage of Ireland's total population."

Three failure scenarios are injectable via tool configuration.
"""

import json
import time
import uuid
from typing import Optional

from anthropic import Anthropic

from .tools import MockWebSearch, MockCalculator, MockMemoryStore, ToolBehaviour
from detector.models import Span

client = Anthropic()

SYSTEM_PROMPT = """You are a research assistant. Use the available tools to answer the user's question.
Think step by step. When you have the final answer, call the finish tool.
Available tools: web_search, calculator, memory_store_write, memory_store_read, finish.
"""


def _make_span(
    op: str,
    input_text: str,
    output_text: str,
    trace_id: str,
    parent_span_id: Optional[str] = None,
) -> Span:
    return Span(
        span_id=str(uuid.uuid4())[:8],
        op=op,
        input=input_text,
        output=output_text,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        timestamp=time.time(),
    )


def run_agent(
    task: str,
    search_behaviour: ToolBehaviour = ToolBehaviour.NORMAL,
    calc_behaviour: ToolBehaviour = ToolBehaviour.NORMAL,
    max_steps: int = 12,
    trace_id: Optional[str] = None,
) -> tuple[list[Span], str]:
    """
    Run the demo agent on a task. Returns (spans, final_answer).

    Args:
        task:             The user's question
        search_behaviour: How web_search should behave
        calc_behaviour:   How calculator should behave
        max_steps:        Hard cap to prevent infinite loops
        trace_id:         Optional trace ID (auto-generated if None)

    Returns:
        spans:        Ordered list of Span objects emitted during execution
        final_answer: The agent's answer string (or empty if interrupted)
    """
    if trace_id is None:
        trace_id = str(uuid.uuid4())[:12]

    search = MockWebSearch(search_behaviour)
    calc = MockCalculator(calc_behaviour)
    memory = MockMemoryStore()

    spans: list[Span] = []
    messages = []
    final_answer = ""

    # Root span
    root_span = _make_span("agent_run", task, "", trace_id)
    spans.append(root_span)

    tools_schema = [
        {
            "name": "web_search",
            "description": "Search the web for factual information",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "calculator",
            "description": "Evaluate a mathematical expression",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
        {
            "name": "memory_store_write",
            "description": "Store a value in memory",
            "input_schema": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        },
        {
            "name": "memory_store_read",
            "description": "Read a value from memory",
            "input_schema": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
        {
            "name": "finish",
            "description": "Return the final answer to the user",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    ]

    messages.append({"role": "user", "content": task})

    for step in range(max_steps):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=tools_schema,
            messages=messages,
        )

        # Collect assistant message
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Emit a span for the LLM reasoning step
        reasoning_text = " ".join(
            block.text for block in assistant_content if hasattr(block, "text")
        )
        reasoning_span = _make_span(
            op="llm_reasoning",
            input_text=task if step == 0 else f"step_{step}",
            output_text=reasoning_text[:500],
            trace_id=trace_id,
            parent_span_id=root_span.span_id,
        )
        spans.append(reasoning_span)

        if response.stop_reason == "end_turn":
            break

        # Process tool calls
        tool_results = []
        for block in assistant_content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input

            # Dispatch to mock tool
            if tool_name == "web_search":
                result = search(tool_input.get("query", ""))
                output = result.content

            elif tool_name == "calculator":
                result = calc(tool_input.get("expression", ""))
                output = result.content

            elif tool_name == "memory_store_write":
                result = memory.write(tool_input["key"], tool_input["value"])
                output = result.content

            elif tool_name == "memory_store_read":
                result = memory.read(tool_input["key"])
                output = result.content

            elif tool_name == "finish":
                final_answer = tool_input.get("answer", "")
                finish_span = _make_span(
                    op="finish",
                    input_text=str(tool_input),
                    output_text=final_answer,
                    trace_id=trace_id,
                    parent_span_id=root_span.span_id,
                )
                spans.append(finish_span)
                root_span.output = final_answer
                return spans, final_answer

            else:
                output = f"Unknown tool: {tool_name}"

            # Emit tool span
            tool_span = _make_span(
                op=tool_name,
                input_text=str(tool_input),
                output_text=output,
                trace_id=trace_id,
                parent_span_id=reasoning_span.span_id,
            )
            spans.append(tool_span)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    # Reached max_steps without finish
    root_span.output = "MAX_STEPS_REACHED"
    return spans, final_answer
