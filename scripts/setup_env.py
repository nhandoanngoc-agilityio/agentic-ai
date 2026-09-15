"""One-shot setup script: install dependencies, then validate the environment.

Installs the project with its `dev` extra, then fails fast — with a clear,
itemized message instead of a cryptic error deep in a pipeline — if any
critical package doesn't actually import, or if `.env` isn't configured
for the selected LLM_PROVIDER.

    python scripts/setup_env.py                # install + validate
    python scripts/setup_env.py --skip-install  # validate only (e.g. CI, after a separate install)
    python scripts/setup_env.py --prod          # also install/validate the `prod` extra (Postgres)
"""

import argparse
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# (module to import, human-readable package name)
_REQUIRED_IMPORTS = [
    ("langgraph.graph", "langgraph"),
    ("langgraph.checkpoint.sqlite", "langgraph-checkpoint-sqlite"),
    ("langchain_core.messages", "langchain-core"),
    ("langchain_anthropic", "langchain-anthropic"),
    ("langchain_openai", "langchain-openai"),
    ("langchain_chroma", "langchain-chroma"),
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
    ("langchain_huggingface", "langchain-huggingface"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("mcp", "mcp"),
    ("langchain_mcp_adapters.client", "langchain-mcp-adapters"),
    ("pypdf", "pypdf"),
    ("tiktoken", "tiktoken"),
    ("pydantic", "pydantic"),
    ("pydantic_settings", "pydantic-settings"),
    ("dotenv", "python-dotenv"),
    ("market_research_team", "market-research-analyst-team (this project, editable install)"),
]

_PROD_IMPORTS = [
    ("langgraph.checkpoint.postgres", "langgraph-checkpoint-postgres"),
    ("psycopg", "psycopg"),
]

_DEV_COMMANDS = ["pytest", "ruff"]


def _install(extra: str) -> None:
    print(f"Installing project (extras: {extra})...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", f".[{extra}]"],
        cwd=_PROJECT_ROOT,
        check=True,
    )


def _check_imports(specs: list[tuple[str, str]]) -> list[str]:
    failures = []
    for module_name, package_label in specs:
        importlib.invalidate_caches()
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            failures.append(f"{package_label}: `import {module_name}` failed ({exc})")
    return failures


def _check_dev_commands(commands: list[str]) -> list[str]:
    return [
        f"`{command}` not found on PATH" for command in commands if shutil.which(command) is None
    ]


def _missing_env_problems(
    env_path: Path, llm_provider: str, process_env: Mapping[str, str]
) -> list[str]:
    if not env_path.exists():
        return [f".env not found at {env_path} — copy .env.example to .env and set your API key"]

    dotenv_pairs = dotenv_values(env_path)
    key_name = "ANTHROPIC_API_KEY" if llm_provider == "anthropic" else "OPENAI_API_KEY"
    has_key = bool(process_env.get(key_name)) or bool(dotenv_pairs.get(key_name))
    if not has_key:
        return [f"{key_name} is not set in .env (required because LLM_PROVIDER={llm_provider})"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip-install", action="store_true", help="Skip `pip install`; only validate."
    )
    parser.add_argument(
        "--prod", action="store_true", help="Also install/validate the `prod` extra."
    )
    args = parser.parse_args()

    if not args.skip_install:
        extra = "dev,prod" if args.prod else "dev"
        try:
            _install(extra)
        except subprocess.CalledProcessError as exc:
            sys.exit(f"pip install failed (exit code {exc.returncode}) — see output above.")

    print("\nValidating critical package imports...")
    import_specs = list(_REQUIRED_IMPORTS) + (_PROD_IMPORTS if args.prod else [])
    problems = _check_imports(import_specs)
    problems += _check_dev_commands(_DEV_COMMANDS)

    print("Validating .env configuration...")
    if importlib.util.find_spec("market_research_team") is not None:
        from market_research_team.config import settings

        problems += _missing_env_problems(_PROJECT_ROOT / ".env", settings.llm_provider, os.environ)
    else:
        problems.append("market_research_team is not importable — cannot check .env configuration")

    if problems:
        print("\nEnvironment validation FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    print("\nEnvironment OK: all critical packages import cleanly and .env is configured.")


if __name__ == "__main__":
    main()
