"""Pulling thumbs-down scores and error traces from Langfuse (fake API object)."""

from datetime import UTC, datetime
from types import SimpleNamespace as NS

from market_research_team.feedback import langfuse_source

SINCE = datetime(2026, 9, 1, tzinfo=UTC)


class _FakeApi:
    def __init__(self, scores=(), observations=(), traces=None, fail=False):
        self._scores = list(scores)
        self._obs = list(observations)
        self._traces = traces or {}
        self._fail = fail
        self.scores = NS(get_many=self._get_scores)
        self.observations = NS(get_many=self._get_obs)
        self.trace = NS(get=self._get_trace)
        self.score_calls: list[dict] = []

    def _get_scores(self, **kw):
        if self._fail:
            raise RuntimeError("401 Unauthorized")
        self.score_calls.append(kw)
        page, limit = kw["page"], kw["limit"]
        chunk = self._scores[(page - 1) * limit : page * limit]
        pages = max(1, -(-len(self._scores) // limit))
        return NS(data=chunk, meta=NS(total_pages=pages))

    def _get_obs(self, **kw):
        return NS(data=self._obs, meta=NS(cursor=None))

    def _get_trace(self, trace_id):
        return self._traces[trace_id]


def _score(session, trace="tr1", ts="2026-09-20T10:00:00+00:00"):
    return NS(session_id=session, trace_id=trace, timestamp=datetime.fromisoformat(ts))


def test_thumbs_down_scores_become_evidence_with_objective_from_the_trace():
    api = _FakeApi(
        scores=[_score("t1")],
        traces={"tr1": NS(session_id="t1", metadata={"objective": "Compare Acme"})},
    )

    evidence, warnings = langfuse_source.collect_from_langfuse(SINCE, api=api)

    assert warnings == []
    ev = evidence["t1"]
    assert ev.ratings == ["down"] and ev.objective == "Compare Acme"
    assert ev.langfuse_trace_ids == ["tr1"] and ev.sources == ["langfuse"]
    assert api.score_calls[0]["name"] == "user_feedback"
    assert api.score_calls[0]["value"] == 0 and api.score_calls[0]["operator"] == "="


def test_score_without_session_uses_trace_lookup():
    api = _FakeApi(
        scores=[_score(None, trace="tr7")],
        traces={"tr7": NS(session_id="t7", metadata={"objective": "o"})},
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
    api = _FakeApi(
        observations=obs, traces={"tr2": NS(session_id="t2", metadata={"objective": "p"})}
    )

    evidence, _ = langfuse_source.collect_from_langfuse(SINCE, api=api)

    assert set(evidence) == {"t2"}
    assert evidence["t2"].langfuse_errors == ["Timeout"]


def test_pagination_stops_at_the_cap_with_a_warning(monkeypatch):
    monkeypatch.setattr(langfuse_source, "MAX_ITEMS", 3)
    monkeypatch.setattr(langfuse_source, "_PAGE_SIZE", 2)
    api = _FakeApi(
        scores=[_score(f"t{i}", trace=f"tr{i}") for i in range(5)],
        traces={f"tr{i}": NS(session_id=f"t{i}", metadata={}) for i in range(5)},
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
        def _get_trace(self, trace_id):
            if trace_id == "gone":
                raise RuntimeError("404 Not Found")
            return super()._get_trace(trace_id)

    api = _Api(
        scores=[_score("t1", trace="tr1"), _score(None, trace="gone")],
        traces={"tr1": NS(session_id="t1", metadata={"objective": "o"})},
    )

    evidence, warnings = langfuse_source.collect_from_langfuse(SINCE, api=api)

    assert set(evidence) == {"t1"}
    assert any("gone" in w for w in warnings)


def test_session_score_without_a_trace_gets_its_objective_from_the_session():
    class _Api(_FakeApi):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.trace = NS(get=self._get_trace, list=self._list)

        def _list(self, **kw):
            assert kw["session_id"] == "t1"
            return NS(data=[NS(metadata={}), NS(metadata={"objective": "Compare Acme"})])

    api = _Api(scores=[NS(session_id="t1", trace_id=None, timestamp=None)])

    evidence, _ = langfuse_source.collect_from_langfuse(SINCE, api=api)

    assert evidence["t1"].objective == "Compare Acme"
