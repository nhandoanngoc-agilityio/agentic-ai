# Agent Development Kit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the context, skills, hooks, and subagent layers of the Agent Development Kit to this repo so Claude Code works better and cheaper on it.

**Architecture:** Plain files under `.claude/` plus a root `CLAUDE.md`. Hooks are bash scripts that parse tool JSON from stdin with `python3`. Agents and skills are Markdown with YAML frontmatter. No Python source changes.

**Tech Stack:** Claude Code settings/hooks/agents/skills, bash, python3 (stdlib only), ruff.

**Spec:** `docs/superpowers/specs/2026-09-17-agent-dev-kit-design.md`

## Global Constraints

- Do not modify anything under `src/`, `tests/`, or `scripts/`.
- Do not edit or remove the existing `[teamai]` hook entries in `.claude/settings.json`.
- Hook scripts must exit 0 on allow, exit 2 with a one-line reason on stderr to block. Never exit 1.
- Hook scripts use only bash and `python3` (stdlib). No `jq`.
- Agents' `model` field must be one of `haiku`, `sonnet`, `fable`.
- `CLAUDE.md` stays between 50 and 70 lines.
- Only `.claude/settings.local.json` and `.claude/worktrees/` remain git-ignored under `.claude/`.
- Commit after every task with `git add` on the exact files listed.

---

### Task 1: Un-ignore `.claude/` in git

**Files:**
- Modify: `.gitignore` (last block, the line `.claude/*`)

**Interfaces:**
- Produces: `.claude/` files are trackable by git for every later task.

- [ ] **Step 1: Confirm current state**

Run: `git check-ignore -v .claude/agents/explorer.md`
Expected: prints `.gitignore:NN:.claude/*` (the path is ignored).

- [ ] **Step 2: Replace the ignore line**

Run:
```bash
python3 - <<'EOF'
from pathlib import Path
p = Path(".gitignore")
s = p.read_text()
assert ".claude/*\n" in s
s = s.replace(".claude/*\n", ".claude/settings.local.json\n.claude/worktrees/\n")
p.write_text(s)
EOF
```

- [ ] **Step 3: Verify**

Run:
```bash
git check-ignore -v .claude/agents/explorer.md; echo "exit=$?"
git check-ignore -v .claude/worktrees/x; echo "exit=$?"
git check-ignore -v .claude/settings.local.json; echo "exit=$?"
```
Expected: first prints nothing and `exit=1` (not ignored); second and third print a match and `exit=0`.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: track .claude/ config in git (except local settings and worktrees)"
```

---

### Task 2: Root `CLAUDE.md`

**Files:**
- Create: `CLAUDE.md`

**Interfaces:**
- Produces: the delegation map naming agents `explorer`, `test-runner`, `code-reviewer`, `langgraph-debugger` (created in Task 4) and skills `langgraph-node`, `run-evals`, `mcp-tool`, `rag-tuning` (created in Task 5).

- [ ] **Step 1: Write the file**

Write `CLAUDE.md` with exactly this content:

````markdown
# Market & Competitor Research Analyst Team

Multi-agent LangGraph demo: a supervisor routes between research (RAG), analytics
(Python stat tools), and reporting (writes markdown via a local MCP filesystem
server). Architecture details: `docs/architecture.md`. Design specs:
`docs/superpowers/specs/`, plans: `docs/superpowers/plans/`.

## Commands

```bash
source .venv/bin/activate
pip install -e ".[dev]"                      # or: python scripts/setup_env.py
pytest                                       # hermetic, no API key needed
ruff check src tests scripts && ruff format --check src tests scripts
python scripts/seed_vectorstore.py           # rebuild Chroma index from data/raw/
python scripts/run_graph_cli.py "<objective>" [--thread-id id]   # REAL LLM CALLS
langgraph dev --no-browser                   # LangGraph Studio on :2024
python scripts/run_evals.py [--provider openai] [--compare] [--langsmith]   # REAL LLM CALLS
```

## Conventions (enforced by the code, not optional)

- Every graph node is registered in `src/market_research_team/graph.py` through
  `with_error_boundary(...)` from `guardrails.py`. Never `add_node` a bare function.
- Nodes are `def node(state: AgentState) -> dict[str, Any]` and return a partial
  state update. State schema lives in `state.py`; add fields there first.
- `pytest` is hermetic: fake LLMs, no network, no seeded vector store, no API key.
  A test that needs a real model belongs in the eval suite, not in `tests/`.
- Supervisor routing is in `supervisor/router.py` (`decide_next_step`,
  `route_from_supervisor`). New routes need the conditional-edge map in `graph.py`.
- Config is `config.py` (pydantic-settings). Read `.env.example` for keys; never
  open `.env`.
- Ruff: line length 100, rules E/F/I/UP. A hook auto-formats edited `.py` files.
- Hermetic MCP tests use the SDK's in-process client session (see
  `tests/test_mcp_server.py`), not a subprocess.

## Cost rules

- `run_evals.py` and `run_graph_cli.py` spend real API money. Ask before running.
- Keep this context small: delegate searches to `explorer` and test runs to
  `test-runner`. Only use `code-reviewer` / `langgraph-debugger` when asked or
  when a review or a real bug justifies a Fable-class agent.

## Delegation map

| Need | Use |
|---|---|
| Find where something lives, summarise a module | `explorer` agent (Haiku) |
| Run tests or lint and get only failures back | `test-runner` agent (Sonnet) |
| Review a diff before merging | `code-reviewer` agent (Fable) |
| Routing loop, state shape, checkpoint/resume bug | `langgraph-debugger` agent (Fable) |
| Add or change a graph node / route | `langgraph-node` skill |
| Run or extend evals, change a prompt or model | `run-evals` skill |
| Add a tool to the MCP server | `mcp-tool` skill |
| Change chunking, retrieval, or reranking | `rag-tuning` skill |

## Guardrails in place (hooks)

- Bash: `rm -rf` on repo dirs, force pushes, `git clean -f`, and writes to `.env` are blocked.
- Files: reading or editing `.env` and `data/checkpoints.sqlite` is blocked.
- Edits to `.py` files are auto-run through `ruff format` and `ruff check --fix`.
````

- [ ] **Step 2: Verify line count**

Run: `wc -l CLAUDE.md`
Expected: between 50 and 70.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md project context for Claude Code"
```

---

### Task 3: Hook scripts and registration

**Files:**
- Create: `.claude/hooks/guard-bash.sh`
- Create: `.claude/hooks/protect-secrets.sh`
- Create: `.claude/hooks/ruff-on-edit.sh`
- Create: `.claude/hooks/test_hooks.sh` (the test harness for the three scripts)
- Modify: `.claude/settings.json` (add `PreToolUse`, extend `PostToolUse`)

**Interfaces:**
- Consumes: Claude Code hook stdin JSON with `tool_name` and `tool_input` (`tool_input.command` for Bash, `tool_input.file_path` for file tools).
- Produces: exit code 0 = allow, 2 = block with reason on stderr.

- [ ] **Step 1: Write the failing test harness**

Create `.claude/hooks/test_hooks.sh`:

```bash
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `chmod +x .claude/hooks/test_hooks.sh && .claude/hooks/test_hooks.sh`
Expected: every line prints `FAIL` (scripts don't exist yet) and exit 1.

- [ ] **Step 3: Write `guard-bash.sh`**

```bash
#!/usr/bin/env bash
# PreToolUse hook for Bash: block destructive commands. Exit 2 = block.
set -u
<<<<<<< HEAD
python3 - <<'EOF'
import json, re, sys
try:
    cmd = json.load(sys.stdin).get("tool_input", {}).get("command", "")
=======
HOOK_INPUT="$(cat)" python3 - <<'PY'
import json, os, re, sys
try:
    cmd = json.loads(os.environ.get("HOOK_INPUT", "")).get("tool_input", {}).get("command", "")
>>>>>>> d641672681169dc82e06f188a2b5977f3e528663
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
<<<<<<< HEAD
EOF
=======
PY
>>>>>>> d641672681169dc82e06f188a2b5977f3e528663
```

- [ ] **Step 4: Write `protect-secrets.sh`**

```bash
#!/usr/bin/env bash
# PreToolUse hook for Read/Edit/Write/MultiEdit: block secret files. Exit 2 = block.
set -u
<<<<<<< HEAD
python3 - <<'EOF'
import json, os, sys
try:
    path = json.load(sys.stdin).get("tool_input", {}).get("file_path", "")
=======
HOOK_INPUT="$(cat)" python3 - <<'PY'
import json, os, sys
try:
    path = json.loads(os.environ.get("HOOK_INPUT", "")).get("tool_input", {}).get("file_path", "")
>>>>>>> d641672681169dc82e06f188a2b5977f3e528663
except Exception:
    sys.exit(0)
base = os.path.basename(path)
norm = path.replace("\\", "/")
if base == ".env" or norm.endswith("data/checkpoints.sqlite"):
    print(f"Blocked by .claude/hooks/protect-secrets.sh: {path} holds secrets or runtime state. "
          "Use .env.example for keys.", file=sys.stderr)
    sys.exit(2)
sys.exit(0)
<<<<<<< HEAD
EOF
=======
PY
>>>>>>> d641672681169dc82e06f188a2b5977f3e528663
```

- [ ] **Step 5: Write `ruff-on-edit.sh`**

```bash
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
```

- [ ] **Step 6: Make executable and run the harness**

Run: `chmod +x .claude/hooks/*.sh && .claude/hooks/test_hooks.sh`
Expected: every line `PASS`, exit 0. If `ruff formatted` fails, check `.venv/bin/ruff` exists.

- [ ] **Step 7: Register hooks in settings.json**

Run:
```bash
python3 - <<'EOF'
import json
from pathlib import Path
p = Path(".claude/settings.json")
s = json.loads(p.read_text())
hooks = s.setdefault("hooks", {})

def entry(matcher, cmd, desc):
    return {"matcher": matcher, "hooks": [{"type": "command", "command": cmd}], "description": desc}

pre = hooks.setdefault("PreToolUse", [])
pre.append(entry("Bash", "bash \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/guard-bash.sh",
                 "Block destructive shell commands"))
pre.append(entry("Read|Edit|Write|MultiEdit", "bash \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/protect-secrets.sh",
                 "Block access to .env and checkpoint DB"))
post = hooks.setdefault("PostToolUse", [])
post.append(entry("Edit|Write|MultiEdit", "bash \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/ruff-on-edit.sh",
                  "Auto-format edited Python files with ruff"))
p.write_text(json.dumps(s, indent=2) + "\n")
EOF
python3 -m json.tool .claude/settings.json >/dev/null && echo JSON_OK
grep -c teamai .claude/settings.json
```
Expected: `JSON_OK`, and the teamai count is 12 (6 descriptions + 6 commands), unchanged from before.

- [ ] **Step 8: Commit**

```bash
git add .claude/hooks/guard-bash.sh .claude/hooks/protect-secrets.sh .claude/hooks/ruff-on-edit.sh .claude/hooks/test_hooks.sh .claude/settings.json
git commit -m "feat(claude): add guardrail hooks (bash guard, secrets, ruff auto-format)"
```

---

### Task 4: Subagents

**Files:**
- Create: `.claude/agents/explorer.md`
- Create: `.claude/agents/test-runner.md`
- Create: `.claude/agents/code-reviewer.md`
- Create: `.claude/agents/langgraph-debugger.md`

**Interfaces:**
- Produces: agent names exactly as referenced in `CLAUDE.md` (Task 2).

- [ ] **Step 1: Write `explorer.md`**

```markdown
---
name: explorer
description: Cheap read-only codebase search. Use to find where something lives, list call sites, or summarise a module before the main session edits anything.
model: haiku
tools: Read, Grep, Glob, Bash
---

You are a read-only scout for this LangGraph repo (`src/market_research_team/`).

Rules:
- Never edit, create, or delete files. Bash is for `ls`, `grep`, `git log`, `git grep` only.
- Answer with file paths and line numbers plus short excerpts (under 10 lines each). Never paste whole files.
- Finish with a 3-line summary: what you found, where, and what is still unknown.
- If the question is about graph routing, start at `graph.py` and `supervisor/router.py`. For state, `state.py`. For config, `config.py`.
```

- [ ] **Step 2: Write `test-runner.md`**

```markdown
---
name: test-runner
description: Runs pytest and ruff and reports only failures. Use after any code change instead of running tests in the main session.
model: sonnet
tools: Bash, Read, Grep
---

You run the hermetic test suite for this repo and report concisely.

Procedure:
1. `source .venv/bin/activate` (or use `.venv/bin/pytest` and `.venv/bin/ruff` directly).
2. Run the requested subset, or by default: `pytest -q` then `ruff check src tests scripts`.
3. If everything passes, reply with one line: `all green: <n> passed, ruff clean`.
4. For each failure: test name, the assertion or exception line, and the 5–10 source lines most likely responsible (read them with Read). No full tracebacks.
5. Never edit code. Never run `scripts/run_evals.py` or `scripts/run_graph_cli.py` (they cost API money).
```

- [ ] **Step 3: Write `code-reviewer.md`**

```markdown
---
name: code-reviewer
description: Reviews a diff for correctness and convention violations before merging. Use on demand, not after every edit; it runs on an expensive model.
model: fable
tools: Read, Grep, Glob, Bash
---

You review changes in this LangGraph multi-agent repo. Read `CLAUDE.md` first, then the diff (`git diff main...HEAD` or the range you are given).

Check, in order of severity:
1. Correctness: wrong state updates, routes not in the conditional-edge map, nodes registered without `with_error_boundary`, swallowed `GraphBubbleUp`, un-bounded loops.
2. Security: prompt-injection surfaces (see `docs/superpowers/specs/2026-09-16-security-hardening-design.md`), path traversal in MCP tools, secrets in code.
3. Tests: are new behaviours covered by hermetic tests? Does any test call a real LLM?
4. Conventions from `CLAUDE.md`.

Output: findings ranked most severe first, each with `file:line`, what is wrong, and a concrete failure scenario. Then a one-line verdict. Read-only: never edit files.
```

- [ ] **Step 4: Write `langgraph-debugger.md`**

```markdown
---
name: langgraph-debugger
description: Diagnoses LangGraph routing loops, state-shape bugs, and checkpoint/resume problems. Use when a run misbehaves and the cause is not obvious.
model: fable
tools: Read, Grep, Glob, Bash
---

You debug the graph in `src/market_research_team/`. Follow systematic debugging: reproduce, locate, explain, then propose the minimal fix. Do not edit files.

Where to look:
- Routing: `supervisor/router.py` (`decide_next_step`, `route_from_supervisor`, visit cap) and the edge map in `graph.py`.
- State: `state.py` (`AgentState`, `error`, `report_discarded`).
- Error containment: `guardrails.py` (`with_error_boundary` records `error` and routes back to the supervisor).
- Resume/interrupt: `run_graph` in `graph.py`, `agents/reporting/node.py`, `checkpointing/`.
- Reproduce with the fake-pipeline pattern in `tests/test_supervisor_routing.py`, never with a real LLM.

Output: root cause with `file:line` evidence, the minimal fix as a diff, and the hermetic test that would catch the regression.
```

- [ ] **Step 5: Verify frontmatter**

Run:
```bash
for f in .claude/agents/*.md; do
  python3 - "$f" <<'EOF'
import sys, re
text = open(sys.argv[1]).read()
m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
assert m, sys.argv[1]
fm = dict(l.split(":", 1) for l in m.group(1).splitlines())
for k in ("name", "description", "model", "tools"): assert k in fm, (sys.argv[1], k)
assert fm["model"].strip() in ("haiku", "sonnet", "fable"), sys.argv[1]
print("OK", fm["name"].strip())
EOF
done
```
Expected: four `OK` lines.

- [ ] **Step 6: Commit**

```bash
git add .claude/agents/
git commit -m "feat(claude): add model-pinned subagents (explorer, test-runner, code-reviewer, langgraph-debugger)"
```

---

### Task 5: Skills

**Files:**
- Create: `.claude/skills/langgraph-node/SKILL.md`
- Create: `.claude/skills/run-evals/SKILL.md`
- Create: `.claude/skills/mcp-tool/SKILL.md`
- Create: `.claude/skills/rag-tuning/SKILL.md`

**Interfaces:**
- Produces: skill names exactly as referenced in `CLAUDE.md` (Task 2).

- [ ] **Step 1: Write `langgraph-node/SKILL.md`**

```markdown
---
name: langgraph-node
description: Use when adding or changing a graph node, sub-agent, or supervisor route in this LangGraph project. Covers state fields, error-boundary registration, routing, and the hermetic test pattern.
---

# Adding or changing a graph node

1. **State first.** Add any new fields to `AgentState` in `src/market_research_team/state.py`. Use `NotRequired[...]` for optional fields. If a new route is needed, extend `RouteDecision`.
2. **Node module.** Create `src/market_research_team/agents/<name>/node.py` with
   `def <name>_node(state: AgentState) -> dict[str, Any]` returning only the keys it changes. Keep LLM/tool calls behind a module-level function (e.g. `_run_pipeline`) so tests can monkeypatch it.
3. **Register with the boundary.** In `graph.py`:
   `builder.add_node("<name>", with_error_boundary("<name>", <name>_node))`
   then `builder.add_edge("<name>", "supervisor")`.
4. **Route.** Add `"<name>": "<name>"` to the `add_conditional_edges` map in `graph.py`, and teach `decide_next_step` in `supervisor/router.py` when to pick it (update the routing prompt and `_SupervisorDecision`).
5. **Test hermetically.** Copy the pattern in `tests/test_supervisor_routing.py`: monkeypatch the node module's pipeline function with a fake, build the graph with `build_production_graph(InMemorySaver())`, invoke with a `thread_id`, and assert on the returned state. Also add a node-level test like those in `tests/test_reporting_node.py`. No real LLM calls.
6. **Run** the `test-runner` agent, then `ruff check src tests`.
7. Update the architecture diagram in `README.md` and `docs/architecture.md` if the topology changed.
```

- [ ] **Step 2: Write `run-evals/SKILL.md`**

```markdown
---
name: run-evals
description: Use when running or extending the prompt evaluation suite, after changing a system prompt, or when switching LLM providers or models. Explains prerequisites, flags, results, and cost.
---

# Running the eval suite

**Costs real API calls. Confirm with the user before running.**

Prerequisites:
- `.env` has the key for the chosen provider (check `.env.example` for names; never read `.env` directly).
- Vector store seeded: `python scripts/seed_vectorstore.py` (needed by the full-pipeline case).

Commands:
```bash
python scripts/run_evals.py                   # current LLM_PROVIDER
python scripts/run_evals.py --provider openai
python scripts/run_evals.py --compare          # anthropic and openai side by side
python scripts/run_evals.py --langsmith        # adds LangSmith dataset sync + LLM-judge; needs LANGSMITH_API_KEY
```

Results: non-zero exit on any failure; JSON report in `data/eval_results/<timestamp>.json`. Categories: query_rewrite, supervisor_decision, analytics, reporting, full_pipeline.

Adding a case: append to the matching list in `src/market_research_team/evaluation/golden_dataset.py` (`QueryRewriteCase`, `SupervisorDecisionCase`, `AnalyticsCase`, `ReportingCase`, `FullPipelineCase`). Ground expectations in `data/raw/` documents, not invented numbers. Deterministic checks live in `evaluation/checks.py`; the harness is `evaluation/offline_eval.py`.

Interpreting a failure: the `detail` string names the failed check. A grounded-number failure means the model invented a figure; fix the prompt, not the check.
```

- [ ] **Step 3: Write `mcp-tool/SKILL.md`**

```markdown
---
name: mcp-tool
description: Use when adding a tool to the local MCP filesystem server or exposing an MCP tool to an agent node. Covers server definition, client wiring, path safety, and the in-process test pattern.
---

# Adding an MCP tool

1. **Define it** in `src/market_research_team/mcp_server/fs_server.py` next to `write_report` and `list_reports`:
   ```python
   @mcp_server.tool()
   def my_tool(arg: str) -> str:
       """One-line description shown to the LLM."""
   ```
   Any path argument must go through `_resolve_report_path` (rejects traversal and nested paths and forces `.md`). Never build paths from raw input.
2. **Expose it.** `load_reporting_tools()` in `src/market_research_team/agents/reporting/mcp_client.py` loads every tool the server exposes, so no client change is needed unless a different agent should get it.
3. **Use it** in the node: bind the tools from `load_reporting_tools()` to the model and handle the tool result like `agents/reporting/node.py` does.
4. **Test** in `tests/test_mcp_server.py` using the SDK's in-process client/server session (no subprocess). Add: one happy-path test, one that rejects a bad path, and one for the empty/missing case.
5. Run the `test-runner` agent.
```

- [ ] **Step 4: Write `rag-tuning/SKILL.md`**

```markdown
---
name: rag-tuning
description: Use when changing chunking, embedding, retrieval, or reranking behaviour, or when research findings look irrelevant. Lists the knobs, the code, the reseed step, and the tests.
---

# Tuning the RAG pipeline

Knobs (all in `src/market_research_team/config.py`, overridable via `.env`):
- `section_chunk_size` (2000), `leaf_chunk_size` (800), `leaf_chunk_overlap` (120)
- `embedding_model_name` (`sentence-transformers/all-MiniLM-L6-v2`)
- `reranker_model_name` (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

Code path: `ingestion/loaders.py` → `ingestion/chunking.py` (`chunk_documents`, hierarchical parent/leaf) → `ingestion/index_build.py` → at query time `agents/research/query_rewriter.py` → `agents/research/retriever.py` (`retrieve_for_queries`) → `agents/research/reranker.py` (`rerank`).

Procedure:
1. Change the knob or code.
2. If chunking or embedding changed, reseed: `python scripts/seed_vectorstore.py` (rebuilds `data/vectorstore/` and `data/processed/`).
3. Run hermetic tests: `pytest tests/test_chunking.py tests/test_retriever.py tests/test_reranker.py tests/test_query_rewriter.py`.
4. To judge quality, run the `run-evals` skill's `full_pipeline` category (costs API calls; ask first).
```

- [ ] **Step 5: Verify frontmatter**

Run:
```bash
for f in .claude/skills/*/SKILL.md; do
  python3 - "$f" <<'EOF'
import sys, re, os
text = open(sys.argv[1]).read()
m = re.match(r"^---\n(.*?)\n---\n", text, re.S); assert m, sys.argv[1]
fm = dict(l.split(":", 1) for l in m.group(1).splitlines())
assert fm["name"].strip() == os.path.basename(os.path.dirname(sys.argv[1])), sys.argv[1]
assert len(fm["description"].strip()) > 40, sys.argv[1]
print("OK", fm["name"].strip())
EOF
done
```
Expected: four `OK` lines.

- [ ] **Step 6: Commit**

```bash
git add .claude/skills/
git commit -m "feat(claude): add project skills (langgraph-node, run-evals, mcp-tool, rag-tuning)"
```

---

### Task 6: Final verification

**Files:** none new.

- [ ] **Step 1: Run everything**

```bash
.claude/hooks/test_hooks.sh
python3 -m json.tool .claude/settings.json >/dev/null && echo JSON_OK
git status --short .claude CLAUDE.md
git ls-files .claude CLAUDE.md
.venv/bin/pytest -q 2>&1 | tail -1
```
Expected: all hook checks `PASS`; `JSON_OK`; `git status` shows nothing uncommitted; `git ls-files` lists settings.json, 3 hook scripts, test harness, 4 agents, 4 skills, CLAUDE.md; pytest tail shows all passed (source untouched).

- [ ] **Step 2: Cross-reference names**

```bash
for n in explorer test-runner code-reviewer langgraph-debugger langgraph-node run-evals mcp-tool rag-tuning; do
  grep -q "$n" CLAUDE.md && echo "CLAUDE.md mentions $n" || echo "MISSING $n in CLAUDE.md"
done
```
Expected: eight "mentions" lines.

- [ ] **Step 3: Smoke test in Claude Code**

Start a new Claude Code session in the repo and run `/agents` and `/hooks`. Expected: the four agents and three new hooks are listed. Ask "where is the supervisor visit cap?" and confirm the `explorer` agent is used.
