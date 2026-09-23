import gradio as gr
from gradio_app.app import build


class _FakeCompiledGraph:
    """A stand-in compiled graph: `build()` only needs something to close
    over in its callbacks, never actually invokes it, so a trivial stub
    keeps this test hermetic (no real checkpointer/sqlite/Postgres I/O)."""

    def invoke(self, _state_or_command: object, config: object) -> dict:
        raise AssertionError("build() should not invoke the graph")


def test_build_returns_blocks_without_raising() -> None:
    demo = build(_FakeCompiledGraph())
    assert isinstance(demo, gr.Blocks)
