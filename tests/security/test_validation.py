import pytest

from market_research_team.security.input_validation import validate_objective


def test_validate_objective_strips_and_returns_valid_text() -> None:
    assert validate_objective("  Assess Acme pricing  ") == "Assess Acme pricing"


def test_validate_objective_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_objective("")


def test_validate_objective_rejects_whitespace_only_string() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_objective("   \n\t  ")


def test_validate_objective_rejects_text_over_max_length() -> None:
    with pytest.raises(ValueError, match="exceeds maximum length"):
        validate_objective("x" * 2001)


def test_validate_objective_accepts_text_at_max_length() -> None:
    text = "x" * 2000
    assert validate_objective(text) == text
