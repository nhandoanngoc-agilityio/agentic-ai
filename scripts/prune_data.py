"""Prune production data older than the retention window (dry run by default).

    python scripts/prune_data.py                # show what would be removed (90 days)
    python scripts/prune_data.py --days 30      # a different window
    python scripts/prune_data.py --apply        # harvest failures first, then delete

The app also prunes automatically at start, at most once a day
(AUTO_PRUNE_ENABLED=false turns that off). See docs/security.md -> Retention.
"""

import argparse
import sys
from datetime import UTC, datetime, timedelta

from market_research_team.config import settings
from market_research_team.feedback.retention import prune


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=settings.audit_retention_days)
    parser.add_argument("--apply", action="store_true", help="actually delete")
    args = parser.parse_args(argv)

    cutoff = datetime.now(UTC) - timedelta(days=args.days)
    report = prune(cutoff, apply=args.apply)
    mode = "APPLIED" if report.applied else "DRY RUN (use --apply to delete)"
    print(f"{mode}: older than {cutoff.isoformat(timespec='seconds')}")
    print(f"  audit lines: {report.audit_removed} removed, {report.audit_kept} kept")
    print(
        f"  checkpoint threads: {report.threads_deleted} deleted "
        f"({report.paused_threads_deleted} were paused awaiting approval)"
    )
    print(f"  remembered reviewer notes: {report.memory_notes_removed} removed")
    if report.applied:
        print(f"  failures harvested first: {report.candidates_harvested}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
