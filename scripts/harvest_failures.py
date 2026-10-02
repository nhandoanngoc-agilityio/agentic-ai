"""Harvest production failures into regression candidates (no LLM calls, free).

    python scripts/harvest_failures.py                 # since the last harvest
    python scripts/harvest_failures.py --since 7d      # or 24h, or 2026-09-01
    python scripts/harvest_failures.py --no-langfuse   # audit log only

Candidates land in data/regression_candidates/pending/ (git-ignored). Curate
them with scripts/promote_case.py or the Gradio "Regressions" tab.
"""

import argparse
import re
import sys
from datetime import UTC, datetime, timedelta

from market_research_team.config import settings
from market_research_team.feedback.candidates import CandidateQueue
from market_research_team.feedback.harvest import harvest
from market_research_team.observability import tracing_enabled


def parse_since(value: str) -> datetime:
    match = re.fullmatch(r"(\d+)([dh])", value.strip())
    if match:
        amount, unit = int(match.group(1)), match.group(2)
        delta = timedelta(days=amount) if unit == "d" else timedelta(hours=amount)
        return datetime.now(UTC) - delta
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", default=None, help="7d, 24h or an ISO date")
    parser.add_argument("--no-langfuse", action="store_true", help="read the audit log only")
    args = parser.parse_args(argv)

    try:
        since = parse_since(args.since) if args.since else None
    except ValueError:
        print(f"--since: expected 7d, 24h or an ISO date, got {args.since!r}", file=sys.stderr)
        return 2

    langfuse = None
    if not args.no_langfuse and tracing_enabled():
        from market_research_team.feedback.langfuse_source import collect_from_langfuse

        langfuse = collect_from_langfuse

    queue = CandidateQueue(settings.candidates_dir)
    try:
        report = harvest(queue, audit_path=settings.audit_log_path, since=since, langfuse=langfuse)
    except OSError as exc:
        print(f"Cannot read the audit log: {exc}", file=sys.stderr)
        return 2

    print(
        f"Scanned {report.runs_scanned} runs: {len(report.new)} new, "
        f"{len(report.merged)} merged, {len(report.skipped)} already decided."
    )
    for warning in report.warnings:
        print(f"WARNING {warning}")
    pending, invalid = queue.list_pending()
    for c in pending:
        print(f"  {c.id}  x{c.occurrences:<3} {','.join(c.triggers):<32} {c.objective[:60]}")
    for cid in invalid:
        print(f"  {cid}  INVALID candidate file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
