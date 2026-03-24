"""
Tool registry for the agent loop.

Register plain Python functions as tools. The registry auto-generates
OpenAI-compatible tool schemas from type hints and docstrings.
"""

import inspect
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Callable


class ToolRegistry:
    """Manages tools available to the agent."""

    def __init__(self):
        self._tools: dict[str, Callable] = {}

    def register(self, func: Callable) -> Callable:
        """Decorator: register a function as an agent tool."""
        self._tools[func.__name__] = func
        return func

    def execute(self, name: str, arguments: dict) -> str:
        """Execute a registered tool by name with given arguments."""
        if name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            result = self._tools[name](**arguments)
            return json.dumps(result) if not isinstance(result, str) else result
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def schemas(self) -> list[dict]:
        """Generate OpenAI-compatible tool schemas for all registered tools."""
        schemas = []
        for name, func in self._tools.items():
            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or ""
            params = {}
            required = []
            for pname, param in sig.parameters.items():
                ptype = "string"
                annotation = param.annotation
                if annotation == int:
                    ptype = "integer"
                elif annotation == float:
                    ptype = "number"
                elif annotation == bool:
                    ptype = "boolean"
                params[pname] = {"type": ptype}
                if param.default is inspect.Parameter.empty:
                    required.append(pname)

            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": doc.split("\n")[0] if doc else name,
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": required,
                    },
                },
            })
        return schemas

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())


# ── Built-in tools ──

registry = ToolRegistry()


@registry.register
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports basic arithmetic, sqrt, pow, log, pi, e."""
    allowed = {
        "sqrt": math.sqrt, "pow": pow, "log": math.log,
        "log10": math.log10, "abs": abs, "round": round,
        "pi": math.pi, "e": math.e,
    }
    # Sanitize: only allow digits, operators, parens, dots, and allowed names
    clean = re.sub(r"[a-zA-Z_]+", lambda m: m.group() if m.group() in allowed else "BLOCKED", expression)
    if "BLOCKED" in clean:
        return json.dumps({"error": f"Unsafe expression: {expression}"})
    try:
        result = eval(clean, {"__builtins__": {}}, allowed)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


@registry.register
def read_file(path: str) -> str:
    """Read the contents of a file and return its text."""
    p = Path(path).resolve()
    if not p.exists():
        return json.dumps({"error": f"File not found: {path}"})
    if not p.is_file():
        return json.dumps({"error": f"Not a file: {path}"})
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > 8000:
            text = text[:8000] + f"\n... [truncated, {len(text)} chars total]"
        return json.dumps({"path": str(p), "content": text})
    except Exception as e:
        return json.dumps({"error": str(e)})


@registry.register
def list_directory(path: str) -> str:
    """List files and directories at the given path."""
    p = Path(path).resolve()
    if not p.exists():
        return json.dumps({"error": f"Path not found: {path}"})
    if not p.is_dir():
        return json.dumps({"error": f"Not a directory: {path}"})
    entries = []
    for item in sorted(p.iterdir()):
        entries.append({
            "name": item.name,
            "type": "dir" if item.is_dir() else "file",
            "size": item.stat().st_size if item.is_file() else None,
        })
    return json.dumps({"path": str(p), "entries": entries[:50]})


@registry.register
def current_datetime() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return json.dumps({
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
    })


@registry.register
def word_count(text: str) -> str:
    """Count words, sentences, and characters in a text."""
    words = len(text.split())
    sentences = len(re.split(r'[.!?]+', text.strip())) - 1
    chars = len(text)
    return json.dumps({"words": words, "sentences": max(sentences, 0), "characters": chars})
