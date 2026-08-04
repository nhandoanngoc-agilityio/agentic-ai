"""Manual end-to-end invocation of the Market & Competitor Research Analyst
Team graph, using the production-guardrail safe entrypoint (`run_graph`).

    python scripts/run_graph_cli.py "Assess Acme vs Globex pricing strategy"
    python scripts/run_graph_cli.py "..." --thread-id demo-1  # persist via checkpointer
"""

import argparse

from market_research_team.checkpointing.store import get_checkpointer
from market_research_team.graph import build_production_graph, run_graph
from market_research_team.state import AgentState


def _initial_state(objective: str) -> AgentState:
    return {
        "messages": [],
        "objective": objective,
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objective", help="Research objective to run the team against.")
    parser.add_argument(
        "--thread-id",
        default=None,
        help="If set, persist/resume this run via a SQLite (or Postgres) checkpointer.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=None,
        help="Override the default recursion limit for this run.",
    )
    args = parser.parse_args()

    compiled_graph = None
    if args.thread_id:
        compiled_graph = build_production_graph(get_checkpointer())

    result = run_graph(
        _initial_state(args.objective),
        compiled_graph=compiled_graph,
        thread_id=args.thread_id,
        recursion_limit=args.recursion_limit,
    )

    print(f"next: {result.get('next')}")
    print(f"error: {result.get('error')}")
    print(f"report_path: {result.get('report_path')}")
    print(f"research_findings: {len(result.get('research_findings', []))}")
    print(f"analytics_results: {len(result.get('analytics_results', []))}")
    print("messages:")
    for message in result.get("messages", []):
        name = getattr(message, "name", None) or "?"
        print(f"  [{name}] {message.content}")


if __name__ == "__main__":
    main()
