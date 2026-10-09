# Security and guardrails

How this project keeps the agents safe, useful and auditable, and what it deliberately does
not do. Design decisions are logged in `architecture.md`.

## Principle

One deterministic guardrail per layer, placed where it is cheapest and where nothing
downstream needs to repeat it. Every control is regex or arithmetic: no extra model calls,
no external moderation service, no measurable latency.

## Layers

| Layer | Control | Code | Behaviour on trigger |
|---|---|---|---|
| Input | Length bounds, control-char stripping, prompt-injection and exfiltration denylist | `security/input_validation.py`, run as the first graph node by `security/input_guard.py` | Run ends with `error`, no LLM call, one `input` event |
| Retrieval | Cross-encoder score floor (`RERANK_SCORE_FLOOR`, default -8.0) | `retrieval/reranker.py` | Low-scoring chunks dropped; an off-topic objective (nothing above the floor) ends the run after one Research pass with a "no relevant material" error |
| Retrieval | Injection-pattern scan on every kept chunk | `agents/research/node.py::filter_injected_chunks` | Chunk dropped, one `retrieval` event naming the source file |
| Tool | Analytics tools are fixed math functions, no code execution | `agents/analytics/tools.py` | n/a |
| Tool | Every numeric tool input must appear in the findings (same 1% tolerance, values >= 10) | `agents/analytics/node.py::tool_input_events`, `security/output_filters.py::ungrounded_inputs` | Result kept, one `tool / ungrounded_tool_input` event; its output no longer counts as evidence for the report |
| Prompt | Retrieved findings are fenced as data in both the analytics and reporting prompts, closing tags escaped | `security/fencing.py` | n/a (prevention) |
| Harness | Timeout and SDK retries on every model call; timeout on the MCP write | `llm.py`, `agents/reporting/node.py` | A stalled call fails the step; the error is labelled transient and the run ends cleanly |
| Tool | Every numeric tool input that only a superseded finding states (a newer source on the same company and topic disagrees) | `agents/analytics/node.py::tool_input_events` | Result kept, one `tool / stale_tool_input` event |
| Tool | MCP server writes `.md` only, inside `REPORTS_DIR`, filename <= 128 chars, content <= 256 KB | `mcp_server/fs_server.py` | Tool call errors; nothing written |
| Tool | An existing report with different content is never replaced unless the caller passes `overwrite=true`; each run writes `<objective slug>-<run id>.md`, so re-running an objective cannot replace an approved report | `mcp_server/fs_server.py`, `agents/reporting/node.py::report_filename` | Tool call errors; the existing report is kept |
| Tool | The MCP server process gets an allowlisted environment (PATH, HOME, locale, temp, venv, `REPORTS_DIR`) and the reports directory as its working directory, so neither the environment nor a `.env` in reach hands it API keys, `DATABASE_URL` or Langfuse keys. Every report write goes through the human-approval interrupt; there is no unreviewed write path | `agents/reporting/mcp_client.py::server_environment` | n/a (prevention) |
| Output | PII redaction (email, phone, card, SSN-shaped) | `security/output_filters.py::redact_pii` | Replaced with `[email redacted]` etc.; `output` event |
| Output | Credential scrub (`sk-`, `sk-ant-`, `lsv2_`, `AKIA`, key=value shapes) | `security/output_filters.py::scrub_secrets` | Replaced with `[secret removed]`; `output` event |
| Output | Number grounding: every figure >= 10 must match a number in the findings, or the output of an analytics call whose inputs came from the findings, within 1% | `security/output_filters.py::flag_unverified_numbers` | Figure gets ` [unverified]`; warning shown at the approval prompt |
| Output | Stale figures: a figure only a superseded finding states and that the report doesn't already label as outdated/earlier or with the old date (a bare number exactly equal to a finding's year is read as a date, not a figure) | `security/output_filters.py::flag_unverified_numbers`, `retrieval/evidence.py::superseded` | Figure gets ` [outdated]`; `output / stale_figure` event; like `unverified_numbers`, it earns one automatic self-check redraft |
| Output | Human approval interrupt before any disk write, max 3 review rounds | `agents/reporting/node.py` | Reviewer approves, rejects with feedback, or discards |
| Policy | Append-only audit log (`AUDIT_LOG_PATH`, default `data/audit.jsonl`) | `security/audit.py`, written by the supervisor at FINISH and by the CLI per human decision | One JSON line: objective, route trace, counts, error, guardrail events, decision |
| Policy | Error boundaries on every node, recursion and visit caps | `guardrails.py`, `graph.py`, `agents/supervisor/router.py` | Run degrades to an `error` state instead of crashing or looping |

The shared regexes live in `security/patterns.py`. Tuning a false positive happens there once
and applies to both the objective and the retrieved chunks.

## What is deliberately not here

- **No moderation API or toxicity classifier.** Objectives are short analyst questions. A
  network call per run adds latency and false-positives on ordinary business phrasing
  ("kill the competitor's pricing advantage" is a valid objective and passes).
- **No LLM judge on output.** It would double tokens per report. Grounding and relevance
  judgment lives in the eval suite (`scripts/run_evals.py`), where it runs on demand.
- **No authorization or RBAC.** The demo has no user identity. If one is added, the natural
  hook is a per-user source allowlist applied in `retrieval/retriever.py` on the chunk's
  `source` metadata.

## Data handling and regulations

- The corpus in `data/raw/` is synthetic market copy about fictional vendors. It contains no
  personal data, so GDPR and HIPAA obligations do not currently apply. The PII redaction
  pass exists so that a future real corpus cannot leak personal data into a report by
  accident, not because any is present today.
- What leaves the machine: the objective, retrieved chunk text and computed metrics go to
  the configured LLM provider (Anthropic or OpenAI) and, if enabled, LangSmith tracing. If
  real customer documents are ever ingested, review the provider's data-retention terms
  and consider disabling `LANGCHAIN_TRACING_V2`.
- Long-term memory holds reviewers' rejection notes, which later drafts see. A note is
  checked against the injection patterns (refused if it matches, a `reviewer_note_refused`
  event), PII- and secret-redacted on the full text, then capped at 500 characters before it
  is stored, and reaches the drafting prompt fenced as data. Evals run without the store.
- What stays local: the vector store, checkpoints, reports and the audit log. The audit log
  and checkpoints are pruned automatically (see Retention below); reports are kept until you
  delete them from `reports/`.
- Secrets: `.env` is never read or written by the agents. Credential-shaped strings are
  scrubbed from reports and from audit entries.

## Retention

- `data/audit.jsonl`, checkpoint threads and remembered reviewer notes (`data/memory.sqlite`,
  or the Postgres store) are kept for `AUDIT_RETENTION_DAYS` (default 90). Older audit lines,
  threads (including runs still paused for approval) and notes are deleted.
- Pruning runs automatically when the Gradio app or the CLI starts, at most once
  every 24 h (`AUTO_PRUNE_ENABLED=false` turns it off). `python scripts/prune_data.py`
  shows what would be removed; `--apply` deletes.
- Before deleting, pruning harvests failures in the window into
  `data/regression_candidates/` (git-ignored), so a thumbs-down, rejection, error or
  blocked attempt is never lost unrecorded. Candidates keep their own scrubbed
  copy of the evidence.
- Candidate text is scrubbed with the PII and secret patterns. Promotion to the
  committed `evals/regressions.jsonl` refuses anything that still matches; the
  curator reads the objective before promoting.
- Langfuse data is not pruned here; set retention in Langfuse.

## Operating the guardrails

- **See what fired**: the CLI prints `guardrail_events` at the end of every run and shows
  output warnings before the approve prompt. The audit log has the same events per run.
- **Tune the retrieval floor**: rerun the measurement in the spec against your index
  (`rerank` on relevant vs off-topic queries, local models only) and set
  `RERANK_SCORE_FLOOR` in `.env`.
- **Add a pattern**: append to the relevant list in `security/patterns.py` and add a case to
  `tests/security/test_validation.py` or `test_output_filters.py`, including a benign
  sentence that must still pass.
- **Costs**: none of these controls call a model. The score floor and chunk filter reduce
  prompt tokens rather than adding any.
