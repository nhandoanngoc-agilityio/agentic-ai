#!/usr/bin/env bash
# Exercises the hook scripts with sample stdin JSON. Run from repo root.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
fail=0

check() { # name, script, json, expected_exit
  local name="$1" script="$2" json="$3" want="$4"
  printf '%s' "$json" | bash "$HERE/$script" >/dev/null 2>&1
  local got=$?
  if [ "$got" -eq "$want" ]; then echo "PASS $name"; else echo "FAIL $name (exit $got, want $want)"; fail=1; fi
}

# guard-bash
check "allow pytest"        guard-bash.sh '{"tool_name":"Bash","tool_input":{"command":"pytest -q"}}' 0
check "block rm -rf data"   guard-bash.sh '{"tool_name":"Bash","tool_input":{"command":"rm -rf data"}}' 2
check "block rm -rf ./src"  guard-bash.sh '{"tool_name":"Bash","tool_input":{"command":"rm -rf ./src"}}' 2
check "block rm -fr ~"      guard-bash.sh '{"tool_name":"Bash","tool_input":{"command":"rm -fr ~"}}' 2
check "allow rm -rf tmpdir" guard-bash.sh '{"tool_name":"Bash","tool_input":{"command":"rm -rf /private/tmp/x"}}' 0
check "block force push"    guard-bash.sh '{"tool_name":"Bash","tool_input":{"command":"git push --force origin main"}}' 2
check "block push -f"       guard-bash.sh '{"tool_name":"Bash","tool_input":{"command":"git push -f"}}' 2
check "allow plain push"    guard-bash.sh '{"tool_name":"Bash","tool_input":{"command":"git push origin main"}}' 0
check "block git clean -f"  guard-bash.sh '{"tool_name":"Bash","tool_input":{"command":"git clean -fd"}}' 2
check "block write .env"    guard-bash.sh '{"tool_name":"Bash","tool_input":{"command":"echo KEY=1 > .env"}}' 2
check "block rm .env"       guard-bash.sh '{"tool_name":"Bash","tool_input":{"command":"rm .env"}}' 2
check "allow cat .env.example" guard-bash.sh '{"tool_name":"Bash","tool_input":{"command":"cat .env.example"}}' 0

# protect-secrets
check "block read .env"     protect-secrets.sh '{"tool_name":"Read","tool_input":{"file_path":"/repo/.env"}}' 2
check "allow .env.example"  protect-secrets.sh '{"tool_name":"Read","tool_input":{"file_path":"/repo/.env.example"}}' 0
check "block checkpoints"   protect-secrets.sh '{"tool_name":"Edit","tool_input":{"file_path":"/repo/data/checkpoints.sqlite"}}' 2
check "allow source file"   protect-secrets.sh '{"tool_name":"Edit","tool_input":{"file_path":"/repo/src/x.py"}}' 0

# ruff-on-edit: formats a messy temp file, always exits 0
tmp="$(mktemp -d)/messy.py"
printf 'import sys,os\nx=1\n' > "$tmp"
check "ruff exits 0"        ruff-on-edit.sh "{\"tool_name\":\"Edit\",\"tool_input\":{\"file_path\":\"$tmp\"}}" 0
if grep -q '^x = 1$' "$tmp"; then echo "PASS ruff formatted"; else echo "FAIL ruff formatted"; fail=1; fi
check "ruff ignores non-py" ruff-on-edit.sh '{"tool_name":"Edit","tool_input":{"file_path":"/repo/README.md"}}' 0

exit $fail
