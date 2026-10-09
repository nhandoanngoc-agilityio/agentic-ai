"""Gradio chat UI for the Market & Competitor Research Analyst Team graph.

Runs the compiled graph in-process via `run_graph`/`Command(resume=...)`,
the same pattern `scripts/run_graph_cli.py` uses for the CLI — no separate
API server.
"""

import queue
import threading
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar

import gradio as gr
from langgraph.types import Command
from matplotlib.figure import Figure

from gradio_app.charts import build_comparison_figure
from gradio_app.regressions_tab import add_regressions_tab
from market_research_team.config import settings
from market_research_team.feedback.candidates import scrub_text
from market_research_team.graph import StepCallback, run_graph
from market_research_team.observability import record_feedback_score
from market_research_team.security import audit
from market_research_team.security.input_validation import validate_objective
from market_research_team.state import AgentState, new_run_state

T = TypeVar("T")

# Shown for nodes whose update carries no message of its own.
_STEP_LABELS = {
    "input_guard": "Objective checked.",
    "reporting": "Report drafted and run through the output checks.",
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
    """Render `report_review_node`'s `interrupt()` payload for the approval turn.

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


def describe_step(node: str, update: dict[str, Any]) -> str:
    """One progress line for a finished node: its own message when it has one
    (e.g. "Research complete: 4 queries, ..."), else a short label."""

    messages = update.get("messages") or []
    content = str(getattr(messages[-1], "content", "") or "") if messages else ""
    if content:
        return content
    if update.get("error"):
        return f"{node} failed: {update['error']}"
    return _STEP_LABELS.get(node, f"{node} finished.")


def run_with_progress(run: Callable[[StepCallback], T]) -> Iterator[tuple[list[str], T | None]]:
    """Run `run(on_step)` on one worker thread and yield `(steps, None)` after
    each node, then `(steps, result)` once it returns.

    One thread for the whole run, rather than letting Gradio step a generator
    across its thread pool, keeps per-thread context (Langfuse tracing, the
    audit thread id) intact for the run's duration.
    """

    events: queue.Queue[tuple[str, Any]] = queue.Queue()

    def on_step(node: str, update: dict[str, Any]) -> None:
        events.put(("step", describe_step(node, update)))

    def worker() -> None:
        try:
            events.put(("done", run(on_step)))
        except BaseException as exc:
            events.put(("error", exc))

    threading.Thread(target=worker, name="gradio-graph-run", daemon=True).start()
    steps: list[str] = []
    while True:
        kind, payload = events.get()
        if kind == "step":
            steps.append(payload)
            yield steps, None
        elif kind == "done":
            yield steps, payload
            return
        else:
            raise payload


def progress_turn(steps: list[str], *, running: bool) -> dict[str, Any]:
    """The chat turn listing a run's steps: live while running, kept afterwards."""

    title = "**Working…**" if running else "**Run steps**"
    lines = "\n".join(f"- {step}" for step in steps) or "- Starting…"
    return {"role": "assistant", "content": f"{title}\n\n{lines}"}


def with_steps(history: list[dict[str, Any]], steps: list[str]) -> list[dict[str, Any]]:
    """Insert the finished step list just before the run's final turn."""

    if not steps or not history:
        return history
    return [*history[:-1], progress_turn(steps, running=False), history[-1]]


def decision_turn(decision: dict[str, Any]) -> dict[str, Any]:
    return {"role": "user", "content": f"Decision: {decision}"}


def submit_objective(
    objective: str,
    history: list[dict[str, Any]],
    thread_id: str | None,
    compiled_graph: Any,
    on_step: StepCallback | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None, Figure | None, bool, str, bool]:
    """Run the graph for a fresh objective and render the outcome.

    Returns `(new_history, thread_id, pending_interrupt, chart_figure,
    approval_row_visible, objective, run_concluded)`. `pending_interrupt` is
    the raw `interrupt()` payload from `report_review_node` when the run paused
    for approval, else `None`. `run_concluded` is true whenever the run
    reached an end state (success, discard, or error) rather than pausing
    for approval -- used to show the satisfaction rating row. `on_step` is
    passed to `run_graph` to report progress (see `run_with_progress`).
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
        if objective.strip():  # an empty submission is not an attempt worth harvesting
            audit.record_input_rejection(thread_id, objective, exc)
        new_history.append({"role": "assistant", "content": f"**Run failed:** {exc}"})
        return new_history, thread_id, None, None, False, objective, True

    try:
        result = run_graph(
            new_run_state(objective),
            compiled_graph=compiled_graph,
            thread_id=thread_id,
            on_step=on_step,
        )
    except Exception as exc:
        new_history.append({"role": "assistant", "content": f"**Run failed:** {exc}"})
        return new_history, thread_id, None, None, False, objective, True

    interrupts = result.get("__interrupt__")
    if interrupts:
        pending_interrupt = interrupts[0].value
        new_history.append(
            {"role": "assistant", "content": _render_interrupt_turn(pending_interrupt)}
        )
        return new_history, thread_id, pending_interrupt, None, True, objective, False

    new_history.append({"role": "assistant", "content": _render_result_turn(result)})
    figure = build_comparison_figure(result.get("analytics_results", []))
    return new_history, thread_id, None, figure, False, objective, True


def resolve_interrupt(
    decision: dict[str, Any],
    history: list[dict[str, Any]],
    thread_id: str,
    compiled_graph: Any,
    on_step: StepCallback | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, Figure | None, bool, bool]:
    """Resume a paused run with a human approval/rejection decision.

    `decision` matches the shape `report_review_node`'s `interrupt()` expects:
    `{"approved": True}`, `{"approved": False, "feedback": "..."}`, or
    `{"discard": True}`. `run_concluded` (see `submit_objective`) is the
    last element.
    """

    new_history = [*history, decision_turn(decision)]
    # Same event the CLI writes: reviewer corrections feed the failure harvester.
    audit.record("human_decision", thread_id, decision=decision)

    try:
        result = run_graph(
            Command(resume=decision),
            compiled_graph=compiled_graph,
            thread_id=thread_id,
            on_step=on_step,
        )
    except Exception as exc:
        new_history.append({"role": "assistant", "content": f"**Run failed:** {exc}"})
        return new_history, None, None, False, True

    interrupts = result.get("__interrupt__")
    if interrupts:
        pending_interrupt = interrupts[0].value
        new_history.append(
            {"role": "assistant", "content": _render_interrupt_turn(pending_interrupt)}
        )
        return new_history, pending_interrupt, None, True, False

    new_history.append({"role": "assistant", "content": _render_result_turn(result)})
    figure = build_comparison_figure(result.get("analytics_results", []))
    return new_history, None, figure, False, True


def record_satisfaction_rating(thread_id: str | None, objective: str, rating: str) -> None:
    """Persist a thumbs up/down rating for the just-concluded run.

    Reuses the existing append-only audit log (`security.audit`) rather
    than a new store -- `audit.record` already never raises and scrubs
    secrets, so a UI click can call this without its own error handling.
    """

    audit.record("user_satisfaction", thread_id, objective=objective, rating=rating)
    record_feedback_score(thread_id, rating, scrub_text(objective))


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
        from market_research_team.checkpointing.store import get_checkpointer, get_memory_store
        from market_research_team.graph import build_production_graph

        compiled_graph = build_production_graph(get_checkpointer(), get_memory_store())

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
                    discard_button = gr.Button("Discard", variant="stop")

                with gr.Row(visible=False) as rating_row:
                    gr.Markdown("How was this run?")
                    thumbs_up_button = gr.Button("\U0001f44d")
                    thumbs_down_button = gr.Button("\U0001f44e")

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

            add_regressions_tab()

        thread_state = gr.State(None)
        interrupt_state = gr.State(None)
        objective_state = gr.State("")

        hidden = gr.update(visible=False)

        def on_submit(objective: str, history: list[dict], thread_id: str | None):
            # Generator: Gradio re-renders after each yield, so the "Working…"
            # turn fills in step by step while the run is going.
            pending_view = [*history, {"role": "user", "content": objective}]
            for steps, outcome in run_with_progress(
                lambda on_step: submit_objective(
                    objective, history, thread_id, compiled_graph, on_step=on_step
                )
            ):
                if outcome is None:
                    live = [*pending_view, progress_turn(steps, running=True)]
                    yield live, gr.skip(), None, None, hidden, "", objective, hidden
                    continue
                new_history, new_thread_id, pending, figure, visible, ran_objective, concluded = (
                    outcome
                )
                yield (
                    with_steps(new_history, steps),
                    new_thread_id,
                    pending,
                    figure,
                    gr.update(visible=visible),
                    "",
                    ran_objective,
                    gr.update(visible=concluded),
                )

        def resume(decision: dict[str, Any], history: list[dict], thread_id: str):
            """Shared by Approve / Reject / Discard: resume the paused run with
            progress. Yields (history, pending, figure, approval_row, rating_row)."""

            pending_view = [*history, decision_turn(decision)]
            for steps, outcome in run_with_progress(
                lambda on_step: resolve_interrupt(
                    decision, history, thread_id, compiled_graph, on_step=on_step
                )
            ):
                if outcome is None:
                    live = [*pending_view, progress_turn(steps, running=True)]
                    yield live, None, None, hidden, hidden
                    continue
                new_history, pending, figure, visible, concluded = outcome
                yield (
                    with_steps(new_history, steps),
                    pending,
                    figure,
                    gr.update(visible=visible),
                    gr.update(visible=concluded),
                )

        def on_approve(history: list[dict], thread_id: str, _pending: dict | None):
            yield from resume({"approved": True}, history, thread_id)

        def on_discard(history: list[dict], thread_id: str, _pending: dict | None):
            yield from resume({"discard": True}, history, thread_id)

        def on_reject(history: list[dict], thread_id: str, _pending: dict | None, feedback: str):
            decision = {"approved": False, "feedback": feedback or None}
            for new_history, pending, figure, approval, rating in resume(
                decision, history, thread_id
            ):
                yield new_history, pending, figure, approval, "", rating

        def on_rate(thread_id: str | None, objective: str, rating: str):
            record_satisfaction_rating(thread_id, objective, rating)
            return gr.update(visible=False)

        submit_button.click(
            on_submit,
            inputs=[objective_box, chatbot, thread_state],
            outputs=[
                chatbot,
                thread_state,
                interrupt_state,
                chart,
                approval_row,
                objective_box,
                objective_state,
                rating_row,
            ],
        )
        approve_button.click(
            on_approve,
            inputs=[chatbot, thread_state, interrupt_state],
            outputs=[chatbot, interrupt_state, chart, approval_row, rating_row],
        )
        reject_button.click(
            on_reject,
            inputs=[chatbot, thread_state, interrupt_state, feedback_box],
            outputs=[chatbot, interrupt_state, chart, approval_row, feedback_box, rating_row],
        )
        discard_button.click(
            on_discard,
            inputs=[chatbot, thread_state, interrupt_state],
            outputs=[chatbot, interrupt_state, chart, approval_row, rating_row],
        )
        thumbs_up_button.click(
            lambda thread_id, objective: on_rate(thread_id, objective, "up"),
            inputs=[thread_state, objective_state],
            outputs=[rating_row],
        )
        thumbs_down_button.click(
            lambda thread_id, objective: on_rate(thread_id, objective, "down"),
            inputs=[thread_state, objective_state],
            outputs=[rating_row],
        )

    return demo


if __name__ == "__main__":
    build().launch()
