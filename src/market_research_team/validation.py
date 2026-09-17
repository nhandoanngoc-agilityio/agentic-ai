"""Input validation for text that becomes graph state, applied at agent boundaries."""

_MAX_OBJECTIVE_LENGTH = 2000


def validate_objective(text: str) -> str:
    """Validate and normalize a research objective before it enters graph state.

    Raises ValueError if the objective is empty/whitespace-only or exceeds
    the maximum length; returns the stripped string otherwise.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Research objective must not be empty.")
    if len(cleaned) > _MAX_OBJECTIVE_LENGTH:
        raise ValueError(
            f"Research objective exceeds maximum length of {_MAX_OBJECTIVE_LENGTH} characters."
        )
    return cleaned
