"""
Agentic tool-calling loop on Mistral Small 4.

A while(tool_use) agent that lets the model decide which tools to call
and when to stop. Works with any OpenAI-compatible endpoint: NIM, vLLM,
Mistral La Plateforme, or OpenAI itself.

Usage:
    from agent import Agent
    from tools import registry

    agent = Agent(model="mistral-small-latest", tools=registry)
    result = agent.run("What files are in the current directory?")
    print(result.answer)
    print(result.trace)    # full step-by-step trace for visualization
"""

import json
import time
from dataclasses import dataclass, field
from openai import OpenAI


@dataclass
class ToolCall:
    """A single tool invocation."""
    name: str
    arguments: dict
    result: str
    duration_ms: float


@dataclass
class Step:
    """One turn of the agent loop."""
    turn: int
    role: str                          # "assistant" or "tool"
    content: str | None                # model text (if any)
    tool_calls: list[ToolCall]         # tools invoked this turn
    finish_reason: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass
class AgentResult:
    """Complete result from an agent run."""
    answer: str
    steps: list[Step] = field(default_factory=list)
    total_turns: int = 0
    total_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0

    @property
    def trace(self) -> list[dict]:
        """Flat trace suitable for visualization."""
        events = []
        for step in self.steps:
            if step.content:
                events.append({
                    "type": "model",
                    "turn": step.turn,
                    "content": step.content,
                    "tokens": step.output_tokens,
                    "latency_ms": step.latency_ms,
                })
            for tc in step.tool_calls:
                events.append({
                    "type": "tool_call",
                    "turn": step.turn,
                    "tool": tc.name,
                    "args": tc.arguments,
                })
                events.append({
                    "type": "tool_result",
                    "turn": step.turn,
                    "tool": tc.name,
                    "result": tc.result[:500],
                    "duration_ms": tc.duration_ms,
                })
        return events


class Agent:
    """
    Autonomous tool-calling agent loop.

    The model decides:
      - which tools to call (from the registry)
      - what arguments to pass
      - when to stop (finish_reason == "stop")

    This is NOT prompt chaining (predefined steps). The model controls
    the flow — it can call tools in any order, call the same tool
    multiple times, or stop after zero tool calls.
    """

    def __init__(
        self,
        model: str = "mistral-small-latest",
        tools=None,
        base_url: str | None = None,
        api_key: str | None = None,
        system_prompt: str | None = None,
        max_turns: int = 10,
    ):
        self.model = model
        self.tools = tools
        self.max_turns = max_turns
        self.system_prompt = system_prompt or (
            "You are a helpful assistant with access to tools. "
            "Use tools when they help answer the user's question. "
            "Call tools as needed, then provide a final answer."
        )

        client_kwargs = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        self.client = OpenAI(**client_kwargs)

    def run(self, prompt: str) -> AgentResult:
        """Execute the agent loop until the model stops or max_turns is hit."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        tool_schemas = self.tools.schemas() if self.tools else []

        result = AgentResult(answer="")
        turn = 0

        while turn < self.max_turns:
            turn += 1

            # ── Model call ──
            t0 = time.perf_counter()
            kwargs = dict(model=self.model, messages=messages)
            if tool_schemas:
                kwargs["tools"] = tool_schemas
            response = self.client.chat.completions.create(**kwargs)
            latency = (time.perf_counter() - t0) * 1000

            choice = response.choices[0]
            msg = choice.message
            usage = response.usage

            step = Step(
                turn=turn,
                role="assistant",
                content=msg.content,
                tool_calls=[],
                finish_reason=choice.finish_reason,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_ms=latency,
            )

            # Append assistant message to history
            messages.append(msg.model_dump())

            # ── Check for tool calls ──
            if not msg.tool_calls or choice.finish_reason == "stop":
                result.answer = msg.content or ""
                result.steps.append(step)
                break

            # ── Execute tools ──
            for tc in msg.tool_calls:
                func_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                t1 = time.perf_counter()
                tool_result = self.tools.execute(func_name, args)
                tool_duration = (time.perf_counter() - t1) * 1000

                step.tool_calls.append(ToolCall(
                    name=func_name,
                    arguments=args,
                    result=tool_result,
                    duration_ms=tool_duration,
                ))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

            result.steps.append(step)

        # Aggregate stats
        result.total_turns = turn
        result.total_tool_calls = sum(len(s.tool_calls) for s in result.steps)
        result.total_input_tokens = sum(s.input_tokens for s in result.steps)
        result.total_output_tokens = sum(s.output_tokens for s in result.steps)
        result.total_latency_ms = sum(s.latency_ms for s in result.steps)

        return result
