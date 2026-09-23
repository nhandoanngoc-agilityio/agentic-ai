from gradio_app.charts import build_comparison_figure
from matplotlib.figure import Figure

from market_research_team.state import AnalyticsResult


def _result(metric: str, value: float, entity: str | None) -> AnalyticsResult:
    return {"metric": metric, "value": value, "detail": "", "entity": entity}


def test_returns_none_when_fewer_than_two_entities() -> None:
    results = [_result("mean", 10.0, "Acme"), _result("mean", 20.0, None)]
    assert build_comparison_figure(results) is None


def test_returns_none_for_empty_results() -> None:
    assert build_comparison_figure([]) is None


def test_returns_figure_for_two_entities_sharing_a_metric() -> None:
    results = [
        _result("mean", 10.0, "Acme"),
        _result("mean", 25.0, "Globex"),
        _result("stdev", 1.5, "Acme"),
    ]
    figure = build_comparison_figure(results)
    assert isinstance(figure, Figure)


def test_ignores_metrics_only_present_for_one_entity() -> None:
    # "stdev" only has an Acme value, so it shouldn't error even though it
    # can't be grouped; "mean" has both and should still produce a figure.
    results = [
        _result("mean", 10.0, "Acme"),
        _result("mean", 25.0, "Globex"),
        _result("stdev", 1.5, "Acme"),
    ]
    figure = build_comparison_figure(results)
    assert figure is not None
    assert len(figure.axes) == 1
