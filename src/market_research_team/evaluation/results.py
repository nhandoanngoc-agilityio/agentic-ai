"""The record every eval case produces, shared by the offline, safety and gate code."""

from dataclasses import dataclass


@dataclass
class EvalResult:
    category: str
    case_name: str
    passed: bool
    detail: str
    provider: str
    # 0.0-1.0 LLM-judge score; None when the case has no judge, no judge ran,
    # or the judge failed (metrics.py counts those as unjudged).
    judge_score: float | None = None
    # Which repeat of the suite (0-based) produced this result.
    repeat: int = 0
