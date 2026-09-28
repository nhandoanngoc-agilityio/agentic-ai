from market_research_team.caching import stats
from market_research_team.caching.response_cache import get_cached_response, put_cached_response
from market_research_team.config import settings

_FINDINGS = [{"source": "competitor_acme.md", "content": "x", "relevance_score": 0.9}]
_RESULTS = [{"metric": "mean", "value": 1.0, "detail": "d", "entity": None}]


def _seed_version(persist_dir, version="v1"):
    persist_dir.mkdir(parents=True, exist_ok=True)
    (persist_dir / ".cache_version").write_text(version, encoding="utf-8")


def test_disabled_by_default_get_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(settings, "response_cache_enabled", False)
    persist_dir = tmp_path / "vectorstore"
    _seed_version(persist_dir)

    put_cached_response("objective", _FINDINGS, _RESULTS, persist_dir)

    assert get_cached_response("objective", persist_dir) is None


def test_cold_miss_then_warm_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(settings, "response_cache_enabled", True)
    persist_dir = tmp_path / "vectorstore"
    _seed_version(persist_dir)

    assert get_cached_response("objective", persist_dir) is None

    put_cached_response("objective", _FINDINGS, _RESULTS, persist_dir)
    cached = get_cached_response("objective", persist_dir)

    assert cached is not None
    findings, results = cached
    assert findings == _FINDINGS
    assert results == _RESULTS


def test_exact_match_only_different_objective_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(settings, "response_cache_enabled", True)
    persist_dir = tmp_path / "vectorstore"
    _seed_version(persist_dir)

    put_cached_response("Compare Acme vs Globex", _FINDINGS, _RESULTS, persist_dir)

    assert get_cached_response("Compare Acme vs Globex pricing", persist_dir) is None


def test_hits_and_misses_are_recorded_in_stats(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(settings, "response_cache_enabled", True)
    persist_dir = tmp_path / "vectorstore"
    _seed_version(persist_dir)
    stats.reset()

    get_cached_response("objective", persist_dir)  # miss
    put_cached_response("objective", _FINDINGS, _RESULTS, persist_dir)
    get_cached_response("objective", persist_dir)  # hit

    assert stats.hit_rate("response") == 0.5


def test_version_bump_invalidates_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_db_path", tmp_path / "cache.sqlite")
    monkeypatch.setattr(settings, "response_cache_enabled", True)
    persist_dir = tmp_path / "vectorstore"
    _seed_version(persist_dir, version="v1")
    put_cached_response("objective", _FINDINGS, _RESULTS, persist_dir)

    _seed_version(persist_dir, version="v2")

    assert get_cached_response("objective", persist_dir) is None
