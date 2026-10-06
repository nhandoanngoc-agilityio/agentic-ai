"""Run a coroutine from synchronous code, whether or not an event loop is
already running in the calling thread.

Graph nodes are synchronous (the CLI, Gradio and evals all call `graph.invoke`),
but the MCP client is async. `asyncio.run` works from a plain thread and
raises `RuntimeError` inside a running loop -- e.g. if the graph is ever
driven from an async handler, a notebook, or `graph.ainvoke`, which runs sync
nodes on the loop's thread. In that case the coroutine runs on its own loop
in a worker thread instead.
"""

import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")


def run_coroutine_sync(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="run-coroutine-sync") as pool:
        return pool.submit(asyncio.run, coro).result()
