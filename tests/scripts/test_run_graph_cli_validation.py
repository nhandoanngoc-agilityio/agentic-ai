"""Tests for the CLI entrypoint's objective validation (scripts/run_graph_cli.py)."""

import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_graph_cli.py"


def _run_cli(objective: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), objective],
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_cli_rejects_empty_objective() -> None:
    result = _run_cli("   ")

    assert result.returncode == 2
    assert "must not be empty" in result.stderr


def test_cli_rejects_objective_over_max_length() -> None:
    result = _run_cli("x" * 2001)

    assert result.returncode == 2
    assert "exceeds maximum length" in result.stderr
