"""Gradio chat UI for the Market & Competitor Research Analyst Team graph.

Runs the compiled graph in-process via `run_graph`/`Command(resume=...)`,
the same pattern `scripts/run_graph_cli.py` uses for the CLI — no separate
API server. See `docs/superpowers/specs/2026-09-22-gradio-frontend-design.md`.
"""

import uuid
from pathlib import Path
from typing import Any

import gradio as gr
from langgraph.types import Command
from matplotlib.figure import Figure

from gradio_app.charts import build_comparison_figure
from market_research_team.config import settings
from market_research_team.graph import run_graph
from market_research_team.security.input_validation import validate_objective
from market_research_team.state import AgentState


def _initial_state(objective: str) -> AgentState:
    return {
        "messages": [],
        "objective": objective,
        "next": "research",
        "research_findings": [],
        "analytics_results": [],
        "report_path": None,
        "error": None,
        "report_discarded": False,
        "guardrail_events": [],
    }


def _render_result_turn(result: AgentState) -> str:
    if result.get("error"):
        return f"**Run failed:** {result['error']}"
    if result.get("report_discarded"):
        return "Report discarded."
    if result.get("report_path"):
        return f"Report written to `{result['report_path']}`."
    return "Run finished with no report written."


def _render_interrupt_turn(payload: dict[str, Any]) -> str:
    """Render `reporting_node`'s `interrupt()` payload for the approval turn.

    Surfaces the filename, review round, and (most importantly) any
    guardrail warnings alongside the draft content, so the human approves
    with eyes open instead of just seeing the draft text.
    """

    header = (
        f"Draft for '{payload['filename']}' — "
        f"review round {payload['attempt']}/{payload['max_attempts']}"
    )
    turn = f"{header}\n\n{payload.get('content', '')}"
    warnings = payload.get("warnings")
    if warnings:
        warnings_block = "\n".join(f"- {warning}" for warning in warnings)
        turn += f"\n\n**Guardrail warnings (review before approving):**\n{warnings_block}"
    return turn


def submit_objective(
    objective: str,
    history: list[dict[str, Any]],
    thread_id: str | None,
    compiled_graph: Any,
) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None, Figure | None, bool]:
    """Run the graph for a fresh objective and render the outcome.

    Returns `(new_history, thread_id, pending_interrupt, chart_figure, approval_row_visible)`.
    `pending_interrupt` is the raw `interrupt()` payload from
    `reporting_node` when the run paused for approval, else `None`.
    """

    # Every call to `submit_objective` starts a brand-new run: always mint a
    # fresh thread_id rather than reusing one left over from a prior
    # objective (that would replay stale checkpointed state — see
    # `resolve_interrupt` for the one path that intentionally resumes a
    # thread). The `thread_id` parameter is ignored for a fresh objective.
    del thread_id
    thread_id = str(uuid.uuid4())

    new_history = [*history, {"role": "user", "content": objective}]

    try:
        objective = validate_objective(objective)
    except Exception as exc:
        new_history.append({"role": "assistant", "content": f"**Run failed:** {exc}"})
        return new_history, thread_id, None, None, False

    try:
        result = run_graph(
            _initial_state(objective),
            compiled_graph=compiled_graph,
            thread_id=thread_id,
        )
    except Exception as exc:
        new_history.append({"role": "assistant", "content": f"**Run failed:** {exc}"})
        return new_history, thread_id, None, None, False

    interrupts = result.get("__interrupt__")
    if interrupts:
        pending_interrupt = interrupts[0].value
        new_history.append(
            {"role": "assistant", "content": _render_interrupt_turn(pending_interrupt)}
        )
        return new_history, thread_id, pending_interrupt, None, True

    new_history.append({"role": "assistant", "content": _render_result_turn(result)})
    figure = build_comparison_figure(result.get("analytics_results", []))
    return new_history, thread_id, None, figure, False


def resolve_interrupt(
    decision: dict[str, Any],
    history: list[dict[str, Any]],
    thread_id: str,
    compiled_graph: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, Figure | None, bool]:
    """Resume a paused run with a human approval/rejection decision.

    `decision` matches the shape `reporting_node`'s `interrupt()` expects:
    `{"approved": True}`, `{"approved": False, "feedback": "..."}`, or
    `{"discard": True}`.
    """

    new_history = [*history, {"role": "user", "content": f"Decision: {decision}"}]

    try:
        result = run_graph(
            Command(resume=decision),
            compiled_graph=compiled_graph,
            thread_id=thread_id,
        )
    except Exception as exc:
        new_history.append({"role": "assistant", "content": f"**Run failed:** {exc}"})
        return new_history, None, None, False

    interrupts = result.get("__interrupt__")
    if interrupts:
        pending_interrupt = interrupts[0].value
        new_history.append(
            {"role": "assistant", "content": _render_interrupt_turn(pending_interrupt)}
        )
        return new_history, pending_interrupt, None, True

    new_history.append({"role": "assistant", "content": _render_result_turn(result)})
    figure = build_comparison_figure(result.get("analytics_results", []))
    return new_history, None, figure, False


def list_report_files() -> list[str]:
    """List markdown report filenames in `settings.reports_dir`, sorted."""

    reports_dir = settings.reports_dir
    if not reports_dir.exists():
        return []
    return sorted(path.name for path in reports_dir.glob("*.md"))


def read_report(filename: str) -> str:
    """Read a report's markdown content by filename.

    Mirrors the MCP `fs_server`'s path-safety rules: rejects anything that
    isn't a plain filename inside `settings.reports_dir`.
    """

    if Path(filename).name != filename:
        raise ValueError(f"'{filename}' must be a plain filename with no directory components.")

    reports_dir = settings.reports_dir.resolve()
    candidate = (reports_dir / filename).resolve()
    if not candidate.is_relative_to(reports_dir):
        raise ValueError(f"'{filename}' resolves outside the reports directory.")

    return candidate.read_text(encoding="utf-8")


def build(compiled_graph: Any | None = None) -> gr.Blocks:
    """Assemble the Gradio Blocks app. Called by `scripts/run_gradio.py`.

    `compiled_graph` defaults to `None`, in which case a real production
    graph (backed by the real checkpointer — sqlite or Postgres) is built.
    Tests pass a fake/stub graph instead, so `build()` stays hermetic.
    """

    if compiled_graph is None:
        from market_research_team.checkpointing.store import get_checkpointer
        from market_research_team.graph import build_production_graph

        compiled_graph = build_production_graph(get_checkpointer())

    with gr.Blocks(title="Market & Competitor Research Analyst Team") as demo:
        gr.Markdown("# Market & Competitor Research Analyst Team")

        with gr.Tabs():
            with gr.Tab("Research"):
                chatbot = gr.Chatbot(label="Conversation")
                objective_box = gr.Textbox(
                    label="Research objective",
                    placeholder="Compare Acme vs Globex go-to-market strategy",
                )
                submit_button = gr.Button("Run", variant="primary")
                chart = gr.Plot(label="Comparison chart")

                with gr.Row(visible=False) as approval_row:
                    approve_button = gr.Button("Approve", variant="primary")
                    feedback_box = gr.Textbox(label="Feedback (for reject)", scale=3)
                    reject_button = gr.Button("Reject")

            with gr.Tab("Reports") as reports_tab:
                report_dropdown = gr.Dropdown(label="Report", choices=list_report_files())
                report_preview = gr.Markdown()

                def on_reports_tab_select():
                    return gr.update(choices=list_report_files())

                def on_report_selected(filename: str | None):
                    if not filename:
                        return ""
                    try:
                        return read_report(filename)
                    except (FileNotFoundError, ValueError) as exc:
                        return f"Could not load '{filename}': {exc}"

                reports_tab.select(on_reports_tab_select, outputs=[report_dropdown])
                report_dropdown.change(
                    on_report_selected, inputs=[report_dropdown], outputs=[report_preview]
                )

        thread_state = gr.State(None)
        interrupt_state = gr.State(None)

        def on_submit(objective: str, history: list[dict], thread_id: str | None):
            new_history, new_thread_id, pending, figure, visible = submit_objective(
                objective, history, thread_id, compiled_graph
            )
            return new_history, new_thread_id, pending, figure, gr.update(visible=visible), ""

        def on_approve(history: list[dict], thread_id: str, _pending: dict | None):
            new_history, pending, figure, visible = resolve_interrupt(
                {"approved": True}, history, thread_id, compiled_graph
            )
            return new_history, pending, figure, gr.update(visible=visible)

        def on_reject(history: list[dict], thread_id: str, _pending: dict | None, feedback: str):
            new_history, pending, figure, visible = resolve_interrupt(
                {"approved": False, "feedback": feedback or None},
                history,
                thread_id,
                compiled_graph,
            )
            return new_history, pending, figure, gr.update(visible=visible), ""

        submit_button.click(
            on_submit,
            inputs=[objective_box, chatbot, thread_state],
            outputs=[chatbot, thread_state, interrupt_state, chart, approval_row, objective_box],
        )
        approve_button.click(
            on_approve,
            inputs=[chatbot, thread_state, interrupt_state],
            outputs=[chatbot, interrupt_state, chart, approval_row],
        )
        reject_button.click(
            on_reject,
            inputs=[chatbot, thread_state, interrupt_state, feedback_box],
            outputs=[chatbot, interrupt_state, chart, approval_row, feedback_box],
        )

    return demo


if __name__ == "__main__":
    build().launch()
