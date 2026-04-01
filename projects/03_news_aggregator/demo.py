"""
Demo: AI News Aggregator — weekly digest powered by Mistral Small 4.

The agent autonomously:
  1. Lists available AI news sources
  2. Fetches RSS feeds from major AI labs and publications
  3. Identifies stories across two categories:
     - Announcements: model releases, product launches, funding, policy
     - Insight & Analysis: opinion pieces, trend analyses, benchmark critiques
  4. Fetches full article text for top stories
  5. Produces a markdown digest with publication dates and reference links

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
that captures both major announcements AND notable insight pieces.

Workflow:
1. Call get_date_range to determine the time window.
2. Call list_news_sources to see available feeds.
3. Fetch feeds from ALL sources using fetch_feed (call them one at a time).
4. Review all articles and select 8-12 stories across TWO categories:
   a) **Announcements** (5-7): new model releases, product launches, \
      funding rounds, acquisitions, policy/regulation changes.
   b) **Insight & Analysis** (3-5): opinion pieces, trend analyses, \
      research deep-dives, benchmark critiques, or industry commentary \
      that shape how practitioners think about the field.
5. For the top stories in both categories, call fetch_article_text to get \
   more detail.
6. Compile a digest in markdown with TWO sections. For each entry include:
   - A one-line headline
   - The **publication date** exactly as given in the feed's "published" \
     field (e.g., "Mar 31, 2026"). Do NOT guess or approximate dates.
   - A 2-3 sentence summary explaining why it matters
   - A markdown reference-style link at the bottom of the digest \
     (e.g., [1]: https://example.com "Source Name")
   Use inline numbered references like [Source][1] next to each headline.
7. Save the digest using save_digest.

Focus on substance over hype. The Announcements section should cover what \
changed; the Insight section should cover what it means.

Important: escape dollar signs as \\$ (e.g., \\$122B) so they are not \
interpreted as LaTeX math delimiters in markdown renderers.\
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
