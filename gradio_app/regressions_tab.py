"""Gradio "Regressions" tab: curate harvested production failures.

Handlers are plain functions over a CandidateQueue so they are testable
without Gradio; `add_regressions_tab` only wires them to components.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr

from market_research_team.config import settings
from market_research_team.feedback.candidates import CandidateQueue, PromotionError
from market_research_team.feedback.harvest import harvest
from market_research_team.feedback.regressions import Expectations
from market_research_team.observability import tracing_enabled

_HEADERS = ["id", "triggers", "occurrences", "objective", "last seen"]


def pending_rows(queue: CandidateQueue) -> list[list[str]]:
    pending, invalid = queue.list_pending()
    rows = [
        [
            c.id,
            ", ".join(c.triggers),
            str(c.occurrences),
            c.objective[:80],
            max((e.get("last_seen", "") for e in c.evidence), default=""),
        ]
        for c in pending
    ]
    rows += [[cid, "INVALID file", "", "", ""] for cid in invalid]
    return rows


def candidate_details(queue: CandidateQueue, cid: str) -> tuple[str, dict[str, Any]]:
    try:
        c = queue.load(cid)
    except KeyError:
        return f"No pending candidate {cid!r}.", {}
    lines = [
        f"### {c.id}",
        f"**Objective:** {c.objective}",
        f"**Triggers:** {', '.join(c.triggers)}",
    ]
    for e in c.evidence:
        lines += [
            "",
            f"**Run** `{e['thread_id']}`: agent {e.get('agent_version') or '?'}, "
            f"{e.get('first_seen')} → {e.get('last_seen')}",
            f"- route: {' → '.join(e.get('route_trace') or []) or 'n/a'}",
            f"- error: {e.get('error') or 'none'}",
            f"- decisions: {json.dumps(e.get('decisions') or [])}",
            f"- guardrail events: {json.dumps(e.get('guardrail_events') or [])}",
            f"- ratings: {', '.join(e.get('ratings') or []) or 'none'}",
            f"- Langfuse trace ids: {', '.join(e.get('langfuse_trace_ids') or []) or 'none'}",
        ]
    expected = (
        c.expectations if not c.expectations.is_empty() else Expectations.from_dict(c.suggested)
    )
    form = {
        "must_block": expected.must_block,
        "required_facts": "\n".join(expected.required_facts),
        "forbidden_substrings": "\n".join(expected.forbidden_substrings),
        "min_findings": "" if expected.min_findings is None else str(expected.min_findings),
        "requires_approval": expected.requires_approval,
        "note": c.note,
    }
    return "\n".join(lines), form


def _lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def promote_from_form(
    queue: CandidateQueue,
    regressions_path: Path,
    cid: str,
    must_block: bool,
    required_facts: str,
    forbidden_substrings: str,
    min_findings: str,
    requires_approval: bool,
    note: str,
) -> str:
    try:
        minimum = int(min_findings) if str(min_findings).strip() else None
    except ValueError:
        return "min_findings must be a whole number or empty."
    try:
        c = queue.load(cid)
        c.expectations = Expectations(
            must_block=bool(must_block),
            required_facts=_lines(required_facts),
            forbidden_substrings=_lines(forbidden_substrings),
            min_findings=minimum,
            requires_approval=bool(requires_approval),
        )
        c.note = note or ""
        queue.save(c)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        entry = queue.promote(cid, regressions_path, now=now)
    except (KeyError, PromotionError) as exc:
        return str(exc).strip("'\"")
    return (
        f"Promoted {entry.id} to {regressions_path}. Commit that file to make it part of the gate."
    )


def reject_from_form(queue: CandidateQueue, cid: str, reason: str) -> str:
    if not (reason or "").strip():
        return "Give a reason for rejecting."
    try:
        queue.reject(cid, reason.strip())
    except KeyError as exc:
        return str(exc).strip("'\"")
    return f"Rejected {cid}."


def harvest_now(queue: CandidateQueue, audit_path: Path) -> str:
    langfuse = None
    if tracing_enabled():
        from market_research_team.feedback.langfuse_source import collect_from_langfuse

        langfuse = collect_from_langfuse
    report = harvest(queue, audit_path=audit_path, langfuse=langfuse)
    message = (
        f"Scanned {report.runs_scanned} runs: {len(report.new)} new, "
        f"{len(report.merged)} merged, {len(report.skipped)} already decided."
    )
    return "\n".join([message, *(f"WARNING {w}" for w in report.warnings)])


def add_regressions_tab() -> None:
    """Build the tab. Call inside `gr.Tabs()`."""

    def queue() -> CandidateQueue:
        return CandidateQueue(settings.candidates_dir)

    with gr.Tab("Regressions"):
        status = gr.Markdown()
        with gr.Row():
            harvest_button = gr.Button("Harvest now")
            refresh_button = gr.Button("Refresh")
        table = gr.Dataframe(headers=_HEADERS, interactive=False)
        selected = gr.Textbox(label="Candidate id")
        details = gr.Markdown()
        must_block = gr.Checkbox(label="must_block (input guard must reject it)")
        required = gr.Textbox(label="required_facts (one per line)", lines=3)
        forbidden = gr.Textbox(label="forbidden_substrings (one per line)", lines=3)
        min_findings = gr.Textbox(label="min_findings (empty = no check)")
        requires_approval = gr.Checkbox(label="requires_approval (must ask before writing)")
        note = gr.Textbox(label="note")
        with gr.Row():
            promote_button = gr.Button("Promote", variant="primary")
            reason = gr.Textbox(label="Reject reason", scale=3)
            reject_button = gr.Button("Reject")

        def on_refresh():
            return pending_rows(queue())

        def on_select(evt: gr.SelectData, rows):
            row = evt.index[0]
            cid = rows.iloc[row, 0] if hasattr(rows, "iloc") else rows[row][0]
            markdown, form = candidate_details(queue(), cid)
            return (
                cid,
                markdown,
                form.get("must_block", False),
                form.get("required_facts", ""),
                form.get("forbidden_substrings", ""),
                form.get("min_findings", ""),
                form.get("requires_approval", False),
                form.get("note", ""),
            )

        def on_promote(cid, mb, rf, fs, mf, ra, nt):
            message = promote_from_form(
                queue(), settings.regressions_path, cid, mb, rf, fs, mf, ra, nt
            )
            return message, pending_rows(queue())

        def on_reject(cid, why):
            return reject_from_form(queue(), cid, why), pending_rows(queue())

        def on_harvest():
            return harvest_now(queue(), settings.audit_log_path), pending_rows(queue())

        form_outputs = [
            selected,
            details,
            must_block,
            required,
            forbidden,
            min_findings,
            requires_approval,
            note,
        ]
        refresh_button.click(on_refresh, outputs=table)
        harvest_button.click(on_harvest, outputs=[status, table])
        table.select(on_select, inputs=table, outputs=form_outputs)
        promote_button.click(
            on_promote,
            inputs=[
                selected,
                must_block,
                required,
                forbidden,
                min_findings,
                requires_approval,
                note,
            ],
            outputs=[status, table],
        )
        reject_button.click(on_reject, inputs=[selected, reason], outputs=[status, table])
