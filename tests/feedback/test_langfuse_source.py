"""Pulling thumbs-down scores and error traces from Langfuse (fake API object)."""

from datetime import UTC, datetime
from types import SimpleNamespace as NS

from market_research_team.feedback import langfuse_source

SINCE = datetime(2026, 9, 1, tzinfo=UTC)


class _FakeApi:
    """Shaped like the current SDK: `scores_v3.get_many_v3` (cursor pages) and
    `observations.get_many`, which serves the ERROR scan and, when filtered by
    `trace_id`/`session_id`, the lookups that replaced the legacy traces API."""

    def __init__(self, scores=(), observations=(), traces=None, sessions=None, fail=False):
        self._scores = list(scores)
        self._obs = list(observations)
        self._traces = traces or {}  # trace_id -> (session_id, metadata)
        self._sessions = sessions or {}  # session_id -> [metadata, ...]
        self._fail = fail
        self.scores_v3 = NS(get_many_v3=self._get_scores)
        self.observations = NS(get_many=self._get_obs)
        self.score_calls: list[dict] = []
        self.lookup_calls: list[dict] = []

    def _get_scores(self, **kw):
        if self._fail:
            raise RuntimeError("401 Unauthorized")
        self.score_calls.append(kw)
        start, limit = int(kw["cursor"] or 0), kw["limit"]
        end = start + limit
        return NS(
            data=self._scores[start:end],
            meta=NS(cursor=str(end) if end < len(self._scores) else None),
        )

    def _get_obs(self, **kw):
        if "trace_id" in kw or "session_id" in kw:
            self.lookup_calls.append(kw)
        if "trace_id" in kw:
            session, metadata = self._traces[kw["trace_id"]]
            return NS(data=[NS(session_id=session, metadata=metadata)], meta=NS(cursor=None))
        if "session_id" in kw:
            rows = [
                NS(session_id=kw["session_id"], metadata=m)
                for m in self._sessions.get(kw["session_id"], [])
            ]
            return NS(data=rows, meta=NS(cursor=None))
        return NS(data=self._obs, meta=NS(cursor=None))


def _score(session, trace="tr1", ts="2026-09-20T10:00:00+00:00"):
    """A v3 score: session-level when `session` is set, else attached to `trace`."""

    subject = NS(kind="session", id=session) if session else NS(kind="trace", id=trace)
    return NS(subject=subject, timestamp=datetime.fromisoformat(ts))


def test_thumbs_down_scores_become_evidence_with_objective_from_the_trace():
    api = _FakeApi(
        scores=[_score(None, trace="tr1")],
        traces={"tr1": ("t1", {"objective": "Compare Acme"})},
    )

    evidence, warnings = langfuse_source.collect_from_langfuse(SINCE, api=api)

    assert warnings == []
    ev = evidence["t1"]
    assert ev.ratings == ["down"] and ev.objective == "Compare Acme"
    assert ev.langfuse_trace_ids == ["tr1"] and ev.sources == ["langfuse"]
    call = api.score_calls[0]
    assert call["name"] == "user_feedback" and call["data_type"] == "NUMERIC"
    assert call["value_max"] == 0.0 and "subject" in call["fields"]
    # Trace lookups go through v2 observations, asking for session + metadata.
    assert api.lookup_calls[0]["trace_id"] == "tr1"
    assert "basic" in api.lookup_calls[0]["fields"] and "metadata" in api.lookup_calls[0]["fields"]


def test_observation_scored_run_uses_the_observations_trace():
    observation = NS(kind="observation", id="obs1", trace_id="tr7")
    api = _FakeApi(
        scores=[NS(subject=observation, timestamp=None)],
        traces={"tr7": ("t7", {"objective": "o"})},
    )

    evidence, _ = langfuse_source.collect_from_langfuse(SINCE, api=api)

    assert set(evidence) == {"t7"}


def test_error_observations_become_langfuse_errors_and_eval_threads_are_skipped():
    obs = [
        NS(
            session_id="t2",
            trace_id="tr2",
            status_message="Timeout",
            name="llm",
            start_time=datetime(2026, 9, 21, tzinfo=UTC),
        ),
        NS(
            session_id="eval-x-r0",
            trace_id="tr3",
            status_message="x",
            name="n",
            start_time=datetime(2026, 9, 21, tzinfo=UTC),
        ),
    ]
    api = _FakeApi(observations=obs, traces={"tr2": ("t2", {"objective": "p"})})

    evidence, _ = langfuse_source.collect_from_langfuse(SINCE, api=api)

    assert set(evidence) == {"t2"}
    assert evidence["t2"].langfuse_errors == ["Timeout"]


def test_pagination_stops_at_the_cap_with_a_warning(monkeypatch):
    monkeypatch.setattr(langfuse_source, "MAX_ITEMS", 3)
    monkeypatch.setattr(langfuse_source, "_PAGE_SIZE", 2)
    api = _FakeApi(
        scores=[_score(None, trace=f"tr{i}") for i in range(5)],
        traces={f"tr{i}": (f"t{i}", {}) for i in range(5)},
    )

    evidence, warnings = langfuse_source.collect_from_langfuse(SINCE, api=api)

    assert len(evidence) == 3
    assert any("cap" in w for w in warnings)


def test_langfuse_error_falls_back_to_audit_only():
    evidence, warnings = langfuse_source.collect_from_langfuse(SINCE, api=_FakeApi(fail=True))

    assert evidence == {}
    assert warnings and "Langfuse unavailable" in warnings[0]


def test_one_failing_trace_lookup_does_not_discard_other_evidence():
    class _Api(_FakeApi):
        def _get_obs(self, **kw):
            if kw.get("trace_id") == "gone":
                raise RuntimeError("404 Not Found")
            return super()._get_obs(**kw)

    api = _Api(
        scores=[_score(None, trace="tr1"), _score(None, trace="gone")],
        traces={"tr1": ("t1", {"objective": "o"})},
    )

    evidence, warnings = langfuse_source.collect_from_langfuse(SINCE, api=api)

    assert set(evidence) == {"t1"}
    assert any("gone" in w for w in warnings)


def test_session_score_without_a_trace_gets_its_objective_from_the_session():
    api = _FakeApi(scores=[_score("t1")], sessions={"t1": [{}, {"objective": "Compare Acme"}]})

    evidence, _ = langfuse_source.collect_from_langfuse(SINCE, api=api)

    assert evidence["t1"].objective == "Compare Acme"
    assert api.lookup_calls[0]["session_id"] == "t1"


def test_experiment_scores_are_not_production_runs():
    api = _FakeApi(scores=[NS(subject=NS(kind="experiment", id="exp1"), timestamp=None)])

    evidence, warnings = langfuse_source.collect_from_langfuse(SINCE, api=api)

    assert evidence == {} and warnings == []
