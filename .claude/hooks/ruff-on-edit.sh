#!/usr/bin/env bash
# PostToolUse hook for Edit/Write/MultiEdit: auto-format edited Python files. Always exits 0.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
file="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)"
case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$file" ] || exit 0
if [ -x "$ROOT/.venv/bin/ruff" ]; then RUFF="$ROOT/.venv/bin/ruff"; else RUFF="$(command -v ruff || true)"; fi
[ -n "$RUFF" ] || exit 0
"$RUFF" format --quiet "$file" >/dev/null 2>&1 || true
"$RUFF" check --fix --quiet "$file" >/dev/null 2>&1 || true
exit 0
