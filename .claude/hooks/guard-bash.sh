#!/usr/bin/env bash
# PreToolUse hook for Bash: block destructive commands. Exit 2 = block.
set -u
HOOK_INPUT="$(cat)" python3 - <<'PY'
import json, os, re, sys
try:
    cmd = json.loads(os.environ.get("HOOK_INPUT", "")).get("tool_input", {}).get("command", "")
except Exception:
    sys.exit(0)

protected = r"(/|~|\.|\.\.|data|reports|src|tests|scripts|docs|\.git)"
rules = [
    (rf"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(\./)?{protected}(/\S*)?(\s|$)",
     "rm -rf on a protected path (/, ~, ., data, reports, src, tests, scripts, docs, .git)"),
    (r"\bgit\s+push\b[^\n]*(\s--force(-with-lease)?\b|\s-f\b)", "git push --force"),
    (r"\bgit\s+clean\s+-[a-zA-Z]*f", "git clean -f"),
    (r"(>|>>|\btee\b)\s*\.env(\s|$)", "writing to .env"),
    (r"\b(rm|mv|cp)\s+([^\n]*\s)?\.env(\s|$)", "removing or moving .env"),
]
for pattern, reason in rules:
    if re.search(pattern, cmd):
        print(f"Blocked by .claude/hooks/guard-bash.sh: {reason}. Command: {cmd}", file=sys.stderr)
        sys.exit(2)
sys.exit(0)
PY
