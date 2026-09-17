# Project Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move modules, tests, and the frontend into the reference layout with zero behaviour change.

**Architecture:** `git mv` for every move so history follows files; a python rewrite script for import paths; ruff to re-sort imports; pytest as the oracle.

**Tech Stack:** git, python3, ruff, pytest, npm (frontend tests).

**Spec:** `docs/superpowers/plans/../specs/2026-09-17-project-restructure-design.md`

## Global Constraints

- No source logic edits; only import lines and doc text change.
- Every task ends with `pytest -q` showing 157 passed, 1 skipped, and `ruff check src tests scripts` clean.
- Commit after each task.

---

### Task 1: Package moves + import rewrite

**Files:** moves per spec table; `__init__.py` created in `src/market_research_team/retrieval/` and `src/market_research_team/security/`.

- [ ] Step 1: Baseline: `.venv/bin/pytest -q | tail -1` → `157 passed, 1 skipped`.
- [ ] Step 2: Moves
```bash
P=src/market_research_team
mkdir -p $P/retrieval $P/security $P/agents/supervisor
git mv $P/agents/research/retriever.py $P/retrieval/retriever.py
git mv $P/agents/research/reranker.py $P/retrieval/reranker.py
git mv $P/agents/research/query_rewriter.py $P/retrieval/query_rewriter.py
git mv $P/validation.py $P/security/input_validation.py
git mv $P/supervisor/router.py $P/agents/supervisor/router.py
git mv $P/supervisor/__init__.py $P/agents/supervisor/__init__.py
touch $P/retrieval/__init__.py $P/security/__init__.py
git add $P/retrieval/__init__.py $P/security/__init__.py
```
- [ ] Step 3: Rewrite imports in `src tests scripts` with the five rules from the spec (exact-string replace, longest first). Then fix `from market_research_team.agents.research import X` forms by hand where X is retriever/reranker/query_rewriter, and `from market_research_team import validation` forms.
- [ ] Step 4: `grep -rnE "market_research_team\.(validation|supervisor)|agents\.research\.(retriever|reranker|query_rewriter)|agents\.research import (retriever|reranker|query_rewriter)" src tests scripts` → no output.
- [ ] Step 5: `.venv/bin/ruff check --fix src tests scripts && .venv/bin/ruff format src tests scripts`; `.venv/bin/pytest -q | tail -1` → 157 passed.
- [ ] Step 6: `git add -A src tests scripts && git commit -m "refactor: move retrieval, security, and supervisor into reference layout"`

### Task 2: Group tests

- [ ] Step 1: `git mv` each file into its folder per the spec table (9 folders).
- [ ] Step 2: `.venv/bin/pytest -q | tail -1` → 157 passed. `ruff check tests` clean.
- [ ] Step 3: `git commit -am "refactor(tests): group tests by area"`

### Task 3: Rename ui/ to frontend/ + AGENTS.md

- [ ] Step 1: `git mv ui frontend`; move untracked `ui/.next` and `ui/node_modules` if still present (`mv ui/* ui/.[!.]* frontend/ 2>/dev/null; rmdir ui`).
- [ ] Step 2: Replace `` `ui/` `` and `` `ui/README.md` `` in README.md with `frontend/`. Check `frontend/README.md` and `frontend/package.json` for self-references to `ui/` and update.
- [ ] Step 3: Create `AGENTS.md`:
```markdown
# Agent instructions

All project rules for AI coding agents live in [CLAUDE.md](CLAUDE.md). Read that file first.
```
- [ ] Step 4: `cd frontend && npm test -- --run 2>&1 | tail -5` → all passing.
- [ ] Step 5: `git add -A && git commit -m "refactor: rename ui/ to frontend/, add AGENTS.md"`

### Task 4: Docs and Claude config

- [ ] Step 1: Update README "Project layout" tree to the new structure (retrieval/, security/, agents/supervisor/, tests grouped, frontend/).
- [ ] Step 2: Update paths in `CLAUDE.md` (`supervisor/router.py` → `agents/supervisor/router.py`; add `retrieval/` and `security/` to conventions), `.claude/agents/explorer.md`, `.claude/agents/langgraph-debugger.md`, `.claude/skills/rag-tuning/SKILL.md`, `.claude/skills/langgraph-node/SKILL.md`.
- [ ] Step 3: Append a "2026-09-17 — Folder restructure" entry to `docs/architecture.md` listing the moves.
- [ ] Step 4: `grep -rn "supervisor/router\|agents/research/retriever\|validation.py\|\`ui/" CLAUDE.md README.md .claude docs/architecture.md` → no output.
- [ ] Step 5: `git add -A && git commit -m "docs: update layout references after restructure"`

### Task 5: Final verification

- [ ] `.venv/bin/pytest -q | tail -1`; `.venv/bin/ruff check src tests scripts`; `.venv/bin/python -c "from market_research_team.graph import graph; print('graph ok')"`; `git status --short` empty; `git worktree list` unchanged.
