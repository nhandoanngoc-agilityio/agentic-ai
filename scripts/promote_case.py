"""Curate harvested regression candidates.

    python scripts/promote_case.py list
    python scripts/promote_case.py show <id>
    python scripts/promote_case.py promote <id>     # after editing its expectations
    python scripts/promote_case.py reject <id> --reason "not reproducible"

Edit data/regression_candidates/pending/<id>.json to fill in "expectations",
then promote. Promotion appends to evals/regressions.jsonl -- commit that file.
"""

import argparse
import json
import sys
from datetime import UTC, datetime

from market_research_team.config import settings
from market_research_team.feedback.candidates import CandidateQueue, PromotionError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for name in ("show", "promote"):
        sub.add_parser(name).add_argument("id")
    reject = sub.add_parser("reject")
    reject.add_argument("id")
    reject.add_argument("--reason", required=True)
    args = parser.parse_args(argv)

    queue = CandidateQueue(settings.candidates_dir)
    try:
        if args.command == "list":
            pending, invalid = queue.list_pending()
            for c in pending:
                print(f"{c.id}  x{c.occurrences:<3} {','.join(c.triggers):<32} {c.objective[:60]}")
            for cid in invalid:
                print(f"{cid}  INVALID candidate file")
        elif args.command == "show":
            print(json.dumps(queue.load(args.id).to_dict(), indent=2))
        elif args.command == "promote":
            now = datetime.now(UTC).isoformat(timespec="seconds")
            entry = queue.promote(args.id, settings.regressions_path, now=now)
            print(f"Promoted {entry.id} to {settings.regressions_path}; commit that file.")
        else:
            queue.reject(args.id, args.reason)
            print(f"Rejected {args.id}.")
    except (KeyError, PromotionError) as exc:
        print(str(exc).strip("'\""), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
