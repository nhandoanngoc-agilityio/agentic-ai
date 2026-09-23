"""Launch the Gradio UI for the Market & Competitor Research Analyst Team.

    python scripts/run_gradio.py

Runs the graph in-process (no separate API server) using the same
`build_production_graph(get_checkpointer())` pattern as
`scripts/run_graph_cli.py`, so report-writing runs can pause for human
approval and be resumed across Gradio interactions on the same thread.
"""

from gradio_app.app import build


def main() -> None:
    build().launch()


if __name__ == "__main__":
    main()
