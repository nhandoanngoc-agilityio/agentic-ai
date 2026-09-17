#!/usr/bin/env bash
# PreToolUse hook for Read/Edit/Write/MultiEdit: block secret files. Exit 2 = block.
set -u
HOOK_INPUT="$(cat)" python3 - <<'PY'
import json, os, sys
try:
    path = json.loads(os.environ.get("HOOK_INPUT", "")).get("tool_input", {}).get("file_path", "")
except Exception:
    sys.exit(0)
base = os.path.basename(path)
norm = path.replace("\\", "/")
if base == ".env" or norm.endswith("data/checkpoints.sqlite"):
    print(f"Blocked by .claude/hooks/protect-secrets.sh: {path} holds secrets or runtime state. "
          "Use .env.example for keys.", file=sys.stderr)
    sys.exit(2)
sys.exit(0)
PY
