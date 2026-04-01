"""
Demo: AI News Aggregator — weekly digest powered by Mistral Small 4.

The agent autonomously:
  1. Lists available AI news sources
  2. Fetches RSS feeds from major AI labs and publications
  3. Identifies the most significant stories from the past week
  4. Fetches full article text for top stories
  5. Produces a ranked weekly digest with summaries

Requires NVIDIA_API_KEY env var (or any OpenAI-compatible endpoint).

Usage:
  export NVIDIA_API_KEY=your_key_here

  # Default: weekly digest from all sources
  python projects/03_news_aggregator/demo.py

  # Custom lookback period
  python projects/03_news_aggregator/demo.py --days 3

  # Use local endpoint
  python projects/03_news_aggregator/demo.py --base-url http://localhost:8000/v1
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects" / "01_tool_calling"))
sys.path.insert(0, str(ROOT / "projects" / "03_news_aggregator"))

from agent import Agent
from news_tools import registry


SYSTEM_PROMPT = """\
You are an AI news analyst. Your job is to produce a concise weekly digest \
of the most significant AI breakthroughs, product launches, and major updates.

Workflow:
1. Call get_date_range to determine the time window.
2. Call list_news_sources to see available feeds.
3. Fetch feeds from ALL sources using fetch_feed (call them one at a time).
4. Review all articles and identify the 5-10 most significant stories — \
   prioritize: new model releases, major product launches, breakthrough \
   research papers, significant funding/acquisitions, and policy/regulation changes.
5. For the top stories, call fetch_article_text to get more detail.
6. Compile a ranked digest in markdown format with:
   - A one-line headline per story
   - A 2-3 sentence summary explaining why it matters
   - The source and link
   - A "Why it matters" section for each entry
7. Save the digest using save_digest.

Focus on substance over hype. Prioritize stories that change what builders \
can do (new models, new APIs, new infrastructure) over opinion pieces or \
general industry commentary.\
"""


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="AI News Aggregator demo")
    parser.add_argument("--model", default="mistralai/mistral-small-4-119b-2603")
    parser.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1",
                        help="OpenAI-compatible endpoint (NIM Cloud, vLLM, Mistral API)")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: NVIDIA_API_KEY env var)")
    parser.add_argument("--out", default="output",
                        help="Output directory")
    parser.add_argument("--days", type=int, default=7,
                        help="How many days back to look (default: 7)")
    parser.add_argument("--max-turns", type=int, default=25,
                        help="Max agent turns (default: 25, needs more for multiple feeds)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: Set NVIDIA_API_KEY env var or pass --api-key")
        sys.exit(1)

    print(f"Model:    {args.model}")
    print(f"Endpoint: {args.base_url}")
    print(f"Tools:    {registry.names}")
    print(f"Lookback: {args.days} days")
    print()

    agent = Agent(
        model=args.model,
        tools=registry,
        base_url=args.base_url,
        api_key=api_key,
        system_prompt=SYSTEM_PROMPT,
        max_turns=args.max_turns,
        verbose=True,
    )

    prompt = (
        f"Build a weekly AI news digest covering the last {args.days} days. "
        f"Fetch from all available sources, identify the most significant stories, "
        f"and produce a ranked digest. Save it when done."
    )

    result = agent.run(prompt)

    # Print results
    print("=" * 60)
    print(f"Digest complete.")
    print(f"\nStats:")
    print(f"  Turns:        {result.total_turns}")
    print(f"  Tool calls:   {result.total_tool_calls}")
    print(f"  Input tokens: {result.total_input_tokens:,}")
    print(f"  Output tokens:{result.total_output_tokens:,}")
    print(f"  Total latency:{result.total_latency_ms:,.0f}ms")

    # Save trace
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_file = out_dir / "news-trace.json"
    with open(trace_file, "w") as f:
        json.dump(result.trace, f, indent=2)
    print(f"\nTrace saved → {trace_file}")

    if result.answer:
        print(f"\n{result.answer[:500]}")


if __name__ == "__main__":
    main()
