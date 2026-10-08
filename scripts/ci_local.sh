#!/usr/bin/env bash
# Run the CI job locally before pushing, the way GitHub Actions runs it: a clean copy
# of the repo (no .venv, no .env, no data/), a fresh Python 3.11 venv at
# ./.venv with `.[dev,prod]`, a minimal environment, then scripts/ci_checks.sh.
#
#   scripts/ci_local.sh             # the committed tree (HEAD): what a push sends
#   scripts/ci_local.sh --worktree  # the working tree as-is, uncommitted changes included
#
# Needs python3.11 on PATH (or PYTHON=/path/to/python3.11). Uses uv when it is
# installed (faster), pip otherwise. The copy is deleted on success and kept
# for inspection on failure.
set -euo pipefail

mode="${1:-head}"
python="${PYTHON:-python3.11}"
if ! command -v "$python" >/dev/null; then
  echo "error: $python not found; install Python 3.11 or set PYTHON=/path/to/python3.11" >&2
  exit 2
fi
repo="$(git rev-parse --show-toplevel)"
work="$(mktemp -d "${TMPDIR:-/tmp}/ci-local.XXXXXX")"

echo "==> Copying the $([ "$mode" = --worktree ] && echo "working tree" || echo "committed tree (HEAD)") to $work"
if [ "$mode" = --worktree ]; then
  # Tracked and untracked files, minus anything .gitignore excludes (.venv, .env, data/...).
  (cd "$repo" && git ls-files -co --exclude-standard -z | xargs -0 tar -cf -) | tar -xf - -C "$work"
else
  git -C "$repo" archive HEAD | tar -xf - -C "$work"
fi

cd "$work"
echo "==> Creating .venv with $("$python" --version) and installing .[dev,prod]"
if command -v uv >/dev/null; then
  uv venv -q --python "$python" .venv
  # Linux CI installs CPU-only torch first; macOS wheels are CPU-only already.
  if [ "$(uname)" = Linux ]; then
    uv pip install -q --python .venv torch --index-url https://download.pytorch.org/whl/cpu
  fi
  uv pip install -q --python .venv -e ".[dev,prod]"
else
  "$python" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  if [ "$(uname)" = Linux ]; then
    .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  fi
  .venv/bin/pip install -q -e ".[dev,prod]"
fi

echo "==> Running scripts/ci_checks.sh in a minimal environment"
# env -i: no DATABASE_URL, API keys or Langfuse keys from your shell -- CI has none.
# What a GitHub runner does set is set here too, so code that reads it (e.g.
# versioning.git_sha() and GITHUB_SHA) behaves as it will in CI.
if env -i HOME="$HOME" PATH="$work/.venv/bin:/usr/bin:/bin" VIRTUAL_ENV="$work/.venv" \
  CI=true GITHUB_ACTIONS=true GITHUB_SHA="$(git -C "$repo" rev-parse HEAD)" \
  bash scripts/ci_checks.sh; then
  cd "$repo" && rm -r "$work"
  echo "==> CI checks passed"
else
  echo "==> CI checks FAILED; the copy is kept at $work"
  exit 1
fi
