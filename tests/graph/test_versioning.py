"""Tests for the agent version manifest (`versioning.py`)."""

import pytest

from market_research_team import versioning
from market_research_team.agents.supervisor import router
from market_research_team.config import settings

_SECTIONS = {"instructions", "model", "tools", "knowledge", "memory_safety"}


@pytest.fixture(autouse=True)
def _clear_caches():
    versioning.clear_cache()
    yield
    versioning.clear_cache()


def test_manifest_covers_every_versioned_part_of_the_agent():
    manifest = versioning.build_manifest()

    assert set(manifest["components"]) == _SECTIONS
    assert set(manifest["component_hashes"]) == _SECTIONS
    assert set(manifest["components"]["instructions"]) == {
        "planner",
        "reporting_guidance",
        "supervisor",
        "query_rewriter",
        "analytics",
        "reporting",
    }
    assert manifest["components"]["tools"]["analytics"]["mean"]
    assert set(manifest["components"]["tools"]["mcp_reports"]) == {"write_report", "list_reports"}


def test_manifest_is_deterministic():
    first = versioning.build_manifest()
    versioning.clear_cache()
    second = versioning.build_manifest()

    assert first["fingerprint"] == second["fingerprint"]
    assert len(first["fingerprint"]) == 12


def test_prompt_change_changes_instructions_hash_and_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
):
    before = versioning.build_manifest()

    monkeypatch.setattr(router, "SYSTEM_PROMPT", router.SYSTEM_PROMPT + " Be terse.")
    versioning.clear_cache()
    after = versioning.build_manifest()

    assert after["component_hashes"]["instructions"] != before["component_hashes"]["instructions"]
    assert after["component_hashes"]["tools"] == before["component_hashes"]["tools"]
    assert after["fingerprint"] != before["fingerprint"]


def test_model_parameter_change_changes_model_hash(monkeypatch: pytest.MonkeyPatch):
    before = versioning.build_manifest()

    monkeypatch.setattr(settings, "llm_temperature", 0.0)
    versioning.clear_cache()
    after = versioning.build_manifest()

    assert after["components"]["model"]["temperature"] == 0.0
    assert after["component_hashes"]["model"] != before["component_hashes"]["model"]


def test_reseed_changes_knowledge_hash(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "vectorstore_dir", tmp_path)
    (tmp_path / ".cache_version").write_text("seed-1", encoding="utf-8")
    before = versioning.build_manifest()

    (tmp_path / ".cache_version").write_text("seed-2", encoding="utf-8")
    after = versioning.build_manifest()  # index stamp is read fresh, no clear_cache

    assert after["components"]["knowledge"]["index_version"] == "seed-2"
    assert after["component_hashes"]["knowledge"] != before["component_hashes"]["knowledge"]


def test_agent_version_label_joins_release_and_fingerprint():
    manifest = versioning.build_manifest()

    assert versioning.agent_version() == f"{manifest['release']}+{manifest['fingerprint']}"


def test_git_sha_prefers_ci_environment(monkeypatch: pytest.MonkeyPatch):
    # Cleared explicitly: a GitHub runner always sets GITHUB_SHA, which wins.
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setenv("CI_COMMIT_SHA", "abcdef1234567890")

    assert versioning.git_sha() == "abcdef123456"


def test_git_sha_reads_github_actions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.setenv("GITHUB_SHA", "1234567890abcdef")
    monkeypatch.setenv("CI_COMMIT_SHA", "ignored")

    assert versioning.git_sha() == "1234567890ab"


def test_trace_metadata_is_flat_strings():
    metadata = versioning.trace_metadata()

    assert metadata["agent_version"] == versioning.agent_version()
    assert all(isinstance(value, str) for value in metadata.values())
    assert {"agent_fingerprint", "git_sha", "model", "instructions_hash"} <= set(metadata)


def test_run_graph_stamps_version_on_the_langfuse_trace(monkeypatch: pytest.MonkeyPatch):
    import contextlib

    import langfuse
    from langchain_core.callbacks.base import BaseCallbackHandler

    from market_research_team import graph as graph_module

    captured: dict = {}

    @contextlib.contextmanager
    def fake_propagate_attributes(**kwargs):
        captured.update(kwargs)
        yield

    class FakeGraph:
        def invoke(self, state, config=None):
            return {**state, "report_path": "reports/x.md"}

    monkeypatch.setattr(graph_module, "tracing_enabled", lambda: True)
    monkeypatch.setattr(graph_module, "get_langfuse_handler", lambda: BaseCallbackHandler())
    monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate_attributes)

    graph_module.run_graph({"objective": "Assess Acme"}, compiled_graph=FakeGraph())

    assert captured["version"] == versioning.agent_version()
    assert captured["metadata"]["objective"] == "Assess Acme"
    assert captured["metadata"]["agent_fingerprint"] == versioning.build_manifest()["fingerprint"]
    assert all(isinstance(value, str) for value in captured["metadata"].values())


def test_diff_manifests_is_empty_for_identical_manifests():
    manifest = versioning.build_manifest()

    assert versioning.diff_manifests(manifest, manifest) == []


def test_diff_manifests_names_changed_component_and_key(monkeypatch: pytest.MonkeyPatch):
    before = versioning.build_manifest()
    monkeypatch.setattr(settings, "llm_temperature", 0.0)
    after = versioning.build_manifest()

    lines = versioning.diff_manifests(before, after)

    assert lines[0].startswith("model: ")
    assert "    temperature: None -> 0.0" in lines
    assert not any(line.startswith("tools:") for line in lines)


def test_diff_manifests_tolerates_an_old_manifest_missing_a_component():
    after = versioning.build_manifest()
    before = {"component_hashes": {}, "components": {}}

    lines = versioning.diff_manifests(before, after)

    assert any(line.startswith("instructions: None -> ") for line in lines)


def test_the_topic_keyword_map_is_part_of_the_knowledge_component():
    knowledge = versioning.build_manifest()["components"]["knowledge"]

    assert knowledge["topic_keywords"]


def test_prompts_no_longer_claim_the_documents_are_undated_or_narrow():
    from market_research_team.agents.planner import node as planner
    from market_research_team.retrieval import query_rewriter

    assert "carry no report dates" not in query_rewriter.SYSTEM_PROMPT
    assert "dated documents" in planner.SYSTEM_PROMPT
    assert "do not cover" not in planner.SYSTEM_PROMPT


def test_the_planner_is_told_not_to_tie_sub_questions_to_a_source():
    from market_research_team.agents.planner import node as planner

    assert "never name a document" in planner.SYSTEM_PROMPT
    assert "customer-review digest" not in planner.SYSTEM_PROMPT


def test_the_insight_limit_is_in_the_manifest():
    safety = versioning.build_manifest()["components"]["memory_safety"]

    assert safety["max_insights"] == settings.run_policy.max_insights == 5


def test_the_insight_length_limit_is_in_the_manifest():
    safety = versioning.build_manifest()["components"]["memory_safety"]

    assert safety["insight_max_chars"] == 300
