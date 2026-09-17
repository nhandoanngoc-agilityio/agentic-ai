"""Manual end-to-end invocation of the Market & Competitor Research Analyst
Team graph, using the production-guardrail safe entrypoint (`run_graph`).

The Reporting Agent always pauses for human approval before writing a
report to disk, so this always runs against a checkpointer (SQLite/Postgres
per `.env`) -- there's no way to resume a paused run otherwise.

    python scripts/run_graph_cli.py "Assess Acme vs Globex pricing strategy"
    python scripts/run_graph_cli.py "..." --thread-id demo-1  # resume a specific thread later
"""

import argparse
import uuid
from typing import Any

from langgraph.types import Command

from market_research_team.checkpointing.store import get_checkpointer
from market_research_team.graph import build_production_graph, run_graph
from market_research_team.state import AgentState
from market_research_team.validation import validate_objective


def _initial_state(objective: str) -> AgentState:
    return {
        "messages": [],
        "objective": objective,
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
    }


def _prompt_for_approval(interrupt_value: dict[str, Any]) -> dict[str, Any]:
    print(
        f"\n--- Reporting Agent wants to write '{interrupt_value.get('filename')}' "
        f"(review round {interrupt_value.get('attempt')}/{interrupt_value.get('max_attempts')}) ---"
    )
    print(interrupt_value.get("content", ""))
    print("--- end of draft ---")

    answer = input("Approve this write to disk? [y/N]: ").strip().lower()
    if answer in ("y", "yes"):
        return {"approved": True}

    feedback = input("Feedback for the next draft (optional, Enter to skip): ").strip()
    return {"approved": False, "feedback": feedback or None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objective", help="Research objective to run the team against.")
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Resume this specific thread id; a random one is generated otherwise.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=None,
        help="Override the default recursion limit for this run.",
    )
    args = parser.parse_args()

    try:
        objective = validate_objective(args.objective)
    except ValueError as exc:
        parser.error(str(exc))

    thread_id = args.thread_id or str(uuid.uuid4())
    compiled_graph = build_production_graph(get_checkpointer())

    result = run_graph(
        _initial_state(objective),
        compiled_graph=compiled_graph,
        thread_id=thread_id,
        recursion_limit=args.recursion_limit,
    )

    while result.get("__interrupt__"):
        decision = _prompt_for_approval(result["__interrupt__"][0].value)
        result = run_graph(
            Command(resume=decision),
            compiled_graph=compiled_graph,
            thread_id=thread_id,
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
