#!/usr/bin/env bash
# The checks the CI job runs (.github/workflows/ci.yml calls this script, and so does
# scripts/ci_local.sh), so CI and a local pre-push run cannot drift apart.
# Run from the repo root with the project venv active.
set -euo pipefail

ruff check src tests scripts gradio_app
ruff format --check src tests scripts gradio_app
basedpyright
pytest -q --cov --junitxml=report.xml
