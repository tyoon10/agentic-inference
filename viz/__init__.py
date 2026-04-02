"""
Agentic AI inference stack visualizations.

Modules:
  stack          — Three-layer architecture (governance → serving → models)
  cost           — API vs self-hosted cost crossover
  trace          — Agent loop execution trace (from 01-tool-calling)
  routing        — Hybrid routing decisions scatter (from 02-hybrid-router)
  trace_analysis — Per-turn latency breakdown with bottleneck annotation (from 03-news-aggregator)

Usage:
  python -m viz.stack
  python -m viz.cost
  python -m viz.trace
  python -m viz.routing
  python -m viz.trace_analysis
"""
