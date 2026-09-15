"""Unit tests for the pure validation helpers in scripts/setup_env.py.

Loaded by file path since `scripts/` isn't a package — mirrors how the
script itself is only ever run directly, never imported.
"""

import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "setup_env.py"
_spec = importlib.util.spec_from_file_location("setup_env", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
setup_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(setup_env)


def test_check_imports_passes_for_real_stdlib_module() -> None:
    failures = setup_env._check_imports([("json", "json")])

    assert failures == []


def test_check_imports_reports_missing_module() -> None:
    failures = setup_env._check_imports([("this_module_does_not_exist", "fake-package")])

    assert len(failures) == 1
    assert "fake-package" in failures[0]


def test_check_dev_commands_reports_missing_command() -> None:
    failures = setup_env._check_dev_commands(["this-command-does-not-exist-anywhere"])

    assert len(failures) == 1
    assert "this-command-does-not-exist-anywhere" in failures[0]


def test_check_dev_commands_passes_for_a_command_on_path() -> None:
    failures = setup_env._check_dev_commands(["python3"])

    assert failures == []


def test_missing_env_problems_flags_absent_env_file(tmp_path: Path) -> None:
    problems = setup_env._missing_env_problems(tmp_path / ".env", "anthropic", {})

    assert len(problems) == 1
    assert ".env not found" in problems[0]


def test_missing_env_problems_flags_missing_key_for_selected_provider(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_PROVIDER=openai\nOPENAI_API_KEY=\n", encoding="utf-8")

    problems = setup_env._missing_env_problems(env_path, "openai", {})

    assert len(problems) == 1
    assert "OPENAI_API_KEY" in problems[0]


def test_missing_env_problems_passes_when_key_set_in_env_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_PROVIDER=anthropic\nANTHROPIC_API_KEY=sk-fake\n", encoding="utf-8")

    problems = setup_env._missing_env_problems(env_path, "anthropic", {})

    assert problems == []


def test_missing_env_problems_passes_when_key_set_in_process_env(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_PROVIDER=anthropic\n", encoding="utf-8")

    problems = setup_env._missing_env_problems(
        env_path, "anthropic", {"ANTHROPIC_API_KEY": "sk-fake"}
    )

    assert problems == []
