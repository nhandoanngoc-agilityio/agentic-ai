"""Print the agent version manifest, or diff it against a saved one.

    python scripts/show_agent_manifest.py                 # label + component hashes
    python scripts/show_agent_manifest.py --json          # full manifest as JSON
    python scripts/show_agent_manifest.py --diff old.json # which components changed

`old.json` can be a saved manifest (`--json > old.json`) or an eval results
file from `data/eval_results/`, which embeds the manifest it was run against.
No LLM calls, no network.
"""

import argparse
import json
import sys
from pathlib import Path

from market_research_team.versioning import build_manifest, diff_manifests, git_sha


def _load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("manifest", data) if isinstance(data, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print the full manifest as JSON")
    parser.add_argument("--diff", type=Path, help="compare against a saved manifest or eval file")
    args = parser.parse_args()

    manifest = build_manifest()
    if args.json:
        print(json.dumps(manifest, indent=2))
        return

    print(f"agent_version: {manifest['release']}+{manifest['fingerprint']}  (git {git_sha()})")
    if args.diff is None:
        for name, value in manifest["component_hashes"].items():
            print(f"  {name:<14} {value}")
        return

    old = _load_manifest(args.diff)
    if "component_hashes" not in old:
        print(f"{args.diff} has no manifest (eval files before versioning don't).", file=sys.stderr)
        sys.exit(2)
    print(f"compared with: {old['release']}+{old['fingerprint']}")
    lines = diff_manifests(old, manifest)
    for line in lines:
        print(f"  {line}")
    if not lines:
        print("  no component changed")


if __name__ == "__main__":
    main()
