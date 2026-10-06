import asyncio

import pytest

from market_research_team.async_utils import run_coroutine_sync


async def _answer() -> int:
    await asyncio.sleep(0)
    return 42


def test_runs_a_coroutine_from_plain_sync_code() -> None:
    assert run_coroutine_sync(_answer()) == 42


async def test_runs_a_coroutine_while_an_event_loop_is_running() -> None:
    # The case `asyncio.run` can't handle: it raises inside a running loop.
    coro = _answer()
    with pytest.raises(RuntimeError, match="running event loop"):
        asyncio.run(coro)
    coro.close()

    assert run_coroutine_sync(_answer()) == 42


async def test_propagates_the_coroutines_exception() -> None:
    async def _boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        run_coroutine_sync(_boom())
