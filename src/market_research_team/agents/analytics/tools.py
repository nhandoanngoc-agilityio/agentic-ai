"""Native Python math/stat tools available to the Analytics Agent."""

import statistics

from langchain_core.tools import tool


@tool
def mean(values: list[float]) -> float:
    """Compute the arithmetic mean of a list of numbers."""
    return statistics.fmean(values)


@tool
def median(values: list[float]) -> float:
    """Compute the median of a list of numbers."""
    return statistics.median(values)


@tool
def stdev(values: list[float]) -> float:
    """Compute the sample standard deviation of a list of numbers.

    Returns 0.0 for fewer than two values, since sample standard deviation
    is undefined below that.
    """
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


@tool
def minimum(values: list[float]) -> float:
    """Return the smallest value in a list of numbers."""
    return min(values)


@tool
def maximum(values: list[float]) -> float:
    """Return the largest value in a list of numbers."""
    return max(values)


@tool
def value_range(values: list[float]) -> float:
    """Return the difference between the largest and smallest value."""
    return max(values) - min(values)


@tool
def percent_change(old_value: float, new_value: float) -> float:
    """Compute the percentage change from old_value to new_value."""
    if old_value == 0:
        raise ValueError("old_value must be non-zero to compute a percent change.")
    return ((new_value - old_value) / old_value) * 100


@tool
def compound_growth_rate(start_value: float, end_value: float, periods: float) -> float:
    """Compute the compound growth rate, as a percentage, over a number of periods."""
    if start_value <= 0 or periods <= 0:
        raise ValueError("start_value and periods must both be positive.")
    return ((end_value / start_value) ** (1 / periods) - 1) * 100


ANALYTICS_TOOLS = [
    mean,
    median,
    stdev,
    minimum,
    maximum,
    value_range,
    percent_change,
    compound_growth_rate,
]
