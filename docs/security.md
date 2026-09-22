# Security and guardrails

How this project keeps the agents safe, useful and auditable, and what it deliberately does
not do. Design rationale: `superpowers/specs/2026-09-18-layered-guardrails-design.md`.

## Principle

One deterministic guardrail per layer, placed where it is cheapest and where nothing
downstream needs to repeat it. Every control is regex or arithmetic: no extra model calls,
no external moderation service, no measurable latency.

## Layers

| Layer | Control | Code | Behaviour on trigger |
|---|---|---|---|
| Input | Length bounds, control-char stripping, prompt-injection and exfiltration denylist | `security/input_validation.py`, run as the first graph node by `security/input_guard.py` | Run ends with `error`, no LLM call, one `input` event |
| Retrieval | Cross-encoder score floor (`RERANK_SCORE_FLOOR`, default -8.0) | `retrieval/reranker.py` | Low-scoring chunks dropped; off-topic objectives get empty context |
| Retrieval | Injection-pattern scan on every kept chunk | `agents/research/node.py::filter_injected_chunks` | Chunk dropped, one `retrieval` event naming the source file |
| Tool | Analytics tools are fixed math functions, no code execution | `agents/analytics/tools.py` | n/a |
| Tool | MCP server writes `.md` only, inside `REPORTS_DIR`, filename <= 128 chars, content <= 256 KB | `mcp_server/fs_server.py` | Tool call errors; nothing written |
| Output | PII redaction (email, phone, card, SSN-shaped) | `security/output_filters.py::redact_pii` | Replaced with `[email redacted]` etc.; `output` event |
| Output | Credential scrub (`sk-`, `sk-ant-`, `lsv2_`, `AKIA`, key=value shapes) | `security/output_filters.py::scrub_secrets` | Replaced with `[secret removed]`; `output` event |
| Output | Number grounding: every figure >= 10 must match a research finding or analytics value within 1% | `security/output_filters.py::flag_unverified_numbers` | Figure gets ` [unverified]`; warning shown at the approval prompt |
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
- What stays local: the vector store, checkpoints, reports and the audit log. Retention is
  manual: delete `data/checkpoints.sqlite*`, `data/audit.jsonl` and `reports/` to purge.
- Secrets: `.env` is never read or written by the agents. Credential-shaped strings are
  scrubbed from reports and from audit entries.

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
