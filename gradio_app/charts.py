"""Turns per-entity `AnalyticsResult`s into a grouped bar chart for the UI.

Kept separate from `agents/analytics/` deliberately: charting is a
presentation concern, not something a graph node produces, so the graph
and its hermetic tests stay chart-free. See the design spec's "Chart
rendering" section.
"""

from collections import defaultdict

from matplotlib.figure import Figure

from market_research_team.state import AnalyticsResult


def build_comparison_figure(results: list[AnalyticsResult]) -> Figure | None:
    """Build a grouped bar chart comparing entities across shared metrics.

    Groups `results` by `metric`, keeping only metrics with two or more
    distinct entities so each bar group is an actual comparison. Returns
    `None` when there's nothing to compare (no results, or every entity is
    `None`/unique per metric).
    """

    by_metric: dict[str, dict[str, float]] = defaultdict(dict)
    for result in results:
        entity = result.get("entity")
        if entity is None:
            continue
        by_metric[result["metric"]][entity] = result["value"]

    comparable = {metric: values for metric, values in by_metric.items() if len(values) >= 2}
    if not comparable:
        return None

    entities = sorted({entity for values in comparable.values() for entity in values})
    metrics = sorted(comparable)

    figure = Figure(figsize=(max(4, len(metrics) * 1.5), 4))
    axis = figure.subplots()
    bar_width = 0.8 / len(entities)
    x_positions = range(len(metrics))

    for index, entity in enumerate(entities):
        offsets = [x + index * bar_width for x in x_positions]
        heights = [comparable[metric].get(entity, 0.0) for metric in metrics]
        axis.bar(offsets, heights, width=bar_width, label=entity)

    axis.set_xticks([x + bar_width * (len(entities) - 1) / 2 for x in x_positions])
    axis.set_xticklabels(metrics, rotation=30, ha="right")
    axis.set_ylabel("Value")
    axis.legend()
    figure.tight_layout()
    return figure
