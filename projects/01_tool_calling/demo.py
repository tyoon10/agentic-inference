"""
Demo: Run the agent loop and visualize the trace.

Requires MISTRAL_API_KEY env var (or any OpenAI-compatible endpoint).
Uses Mistral's La Plateforme by default — no local GPU needed.

Usage:
  # Set your API key
  export MISTRAL_API_KEY=your_key_here

  # Run with default prompt
  python -m projects.01-tool-calling.demo

  # Custom prompt
  python -m projects.01-tool-calling.demo "How many words are in the README?"

  # Use a local NIM/vLLM endpoint instead
  python -m projects.01-tool-calling.demo --base-url http://localhost:8000/v1
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects" / "01_tool_calling"))

from agent import Agent
from tools import registry
from viz.trace import render as render_trace


DEFAULT_PROMPT = (
    "List the files in the current directory, read the README, "
    "count how many words it has, and give me a summary."
)


def main():
    parser = argparse.ArgumentParser(description="Agent loop demo")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="mistral-small-latest")
    parser.add_argument("--base-url", default="https://api.mistral.ai/v1",
                        help="OpenAI-compatible endpoint")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: MISTRAL_API_KEY env var)")
    parser.add_argument("--out", default="output",
                        help="Output directory for trace visualization")
    parser.add_argument("--max-turns", type=int, default=10)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: Set MISTRAL_API_KEY env var or pass --api-key")
        sys.exit(1)

    print(f"Model:    {args.model}")
    print(f"Endpoint: {args.base_url}")
    print(f"Tools:    {registry.names}")
    print(f"Prompt:   {args.prompt[:80]}...")
    print()

    agent = Agent(
        model=args.model,
        tools=registry,
        base_url=args.base_url,
        api_key=api_key,
        max_turns=args.max_turns,
    )

    result = agent.run(args.prompt)

    # Print results
    print("=" * 60)
    print(f"Answer: {result.answer}")
    print(f"\nStats:")
    print(f"  Turns:        {result.total_turns}")
    print(f"  Tool calls:   {result.total_tool_calls}")
    print(f"  Input tokens: {result.total_input_tokens:,}")
    print(f"  Output tokens:{result.total_output_tokens:,}")
    print(f"  Total latency:{result.total_latency_ms:,.0f}ms")

    # Save trace
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_file = out_dir / "trace.json"
    with open(trace_file, "w") as f:
        json.dump(result.trace, f, indent=2)
    print(f"\nTrace saved → {trace_file}")

    # Render visualization
    render_trace(out_dir, result.trace)


if __name__ == "__main__":
    main()
