"""Unit tests for the pure helper in scripts/run_evals.py.

Loaded by file path since `scripts/` isn't a package, same pattern as
tests/test_setup_env.py.
"""

import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_evals.py"
_spec = importlib.util.spec_from_file_location("run_evals", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
run_evals = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_evals)


def test_langsmith_prereq_error_when_flag_set_and_key_missing() -> None:
    assert run_evals._langsmith_prereq_error(True, None) == (
        "--langsmith requires LANGSMITH_API_KEY to be set"
    )


def test_langsmith_prereq_error_none_when_flag_set_and_key_present() -> None:
    assert run_evals._langsmith_prereq_error(True, "sk-fake") is None


def test_langsmith_prereq_error_none_when_flag_not_set() -> None:
    assert run_evals._langsmith_prereq_error(False, None) is None
