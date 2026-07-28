import statistics

import pytest

from market_research_team.agents.analytics.tools import (
    compound_growth_rate,
    maximum,
    mean,
    median,
    minimum,
    percent_change,
    stdev,
    value_range,
)


def test_mean() -> None:
    assert mean.invoke({"values": [1, 2, 3, 4]}) == 2.5


def test_median_odd_and_even_length() -> None:
    assert median.invoke({"values": [1, 3, 2]}) == 2
    assert median.invoke({"values": [1, 2, 3, 4]}) == 2.5


def test_stdev_below_two_values_is_zero() -> None:
    assert stdev.invoke({"values": [5]}) == 0.0


def test_stdev_matches_statistics_module() -> None:
    values = [2, 4, 4, 4, 5, 5, 7, 9]
    assert stdev.invoke({"values": values}) == pytest.approx(statistics.stdev(values))


def test_minimum_and_maximum() -> None:
    assert minimum.invoke({"values": [3, 1, 2]}) == 1
    assert maximum.invoke({"values": [3, 1, 2]}) == 3


def test_value_range() -> None:
    assert value_range.invoke({"values": [10, 2, 7]}) == 8


def test_percent_change() -> None:
    assert percent_change.invoke({"old_value": 50, "new_value": 75}) == pytest.approx(50.0)


def test_percent_change_rejects_zero_old_value() -> None:
    with pytest.raises(ValueError):
        percent_change.invoke({"old_value": 0, "new_value": 10})


def test_compound_growth_rate() -> None:
    result = compound_growth_rate.invoke({"start_value": 100, "end_value": 121, "periods": 2})
    assert result == pytest.approx(10.0)


def test_compound_growth_rate_rejects_non_positive_start() -> None:
    with pytest.raises(ValueError):
        compound_growth_rate.invoke({"start_value": 0, "end_value": 10, "periods": 2})
