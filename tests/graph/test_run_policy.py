"""RunPolicy: every run limit in one place, configurable, read at run time."""

import pytest
from langchain_core.messages import AIMessage

from market_research_team.agents.analytics.node import run_tool_calling_loop
from market_research_team.agents.analytics.tools import mean
from market_research_team.config import RunPolicy, Settings, settings
from market_research_team.versioning import build_manifest


def test_limits_can_be_overridden_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN_POLICY__MAX_ROUTING_VISITS", "10")
    monkeypatch.setenv("RUN_POLICY__MAX_PLAN_ITEMS", "3")

    policy = Settings(_env_file=None).run_policy  # type: ignore[call-arg]

    assert (policy.max_routing_visits, policy.max_plan_items) == (10, 3)
    assert policy.max_review_rounds == RunPolicy().max_review_rounds  # others keep defaults


def test_every_limit_is_in_the_version_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "run_policy", RunPolicy(max_routing_visits=11))

    manifest = build_manifest()

    safety = manifest["components"]["memory_safety"]
    knowledge = manifest["components"]["knowledge"]
    assert safety["max_routing_visits"] == 11
    names = set(RunPolicy.model_fields) - {"max_findings"}
    assert names <= set(safety)
    assert knowledge["max_research_findings"] == settings.run_policy.max_findings


def test_the_analytics_loop_reads_its_limit_at_run_time(monkeypatch: pytest.MonkeyPatch) -> None:
    class _AlwaysCallsATool:
        calls = 0

        def bind_tools(self, _tools: object) -> "_AlwaysCallsATool":
            return self

        def invoke(self, _messages: object, config: object = None) -> AIMessage:
            type(self).calls += 1
            call = {"name": "mean", "args": {"values": [1, 2]}, "id": "c", "type": "tool_call"}
            return AIMessage(content="", tool_calls=[call])

    monkeypatch.setattr(settings, "run_policy", RunPolicy(max_tool_iterations=2))

    run_tool_calling_loop(_AlwaysCallsATool(), [mean], "x", [])  # type: ignore[arg-type]

    assert _AlwaysCallsATool.calls == 2
