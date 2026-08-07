"""Golden evaluation cases, grounded in the sample documents under `data/raw/`.

These reference real facts from the seeded corpus (Acme's $49/seat Starter
price, Globex's $150K-$400K ACV range) rather than synthetic placeholders,
so a failing case signals an actual behavioral regression — a hallucinated
number or a dropped constraint — not a mismatch against a made-up
expectation.
"""

from dataclasses import dataclass, field

from market_research_team.state import AnalyticsResult, ResearchFinding


@dataclass
class QueryRewriteCase:
    name: str
    objective: str
    min_queries: int = 1
    max_queries: int = 6
    required_any_keywords: list[str] = field(default_factory=list)


@dataclass
class SupervisorDecisionCase:
    name: str
    research_findings: list[ResearchFinding]
    analytics_results: list[AnalyticsResult]
    report_path: str | None
    allowed_decisions: tuple[str, ...]


@dataclass
class AnalyticsCase:
    name: str
    objective: str
    findings: list[ResearchFinding]
    min_tool_calls: int = 1
    plausible_values: list[float] = field(default_factory=list)
    tolerance: float = 1.0


@dataclass
class ReportingCase:
    name: str
    objective: str
    findings: list[ResearchFinding]
    results: list[AnalyticsResult]
    required_sections: list[str] = field(
        default_factory=lambda: ["Objective", "Findings", "Analysis"]
    )
    required_facts: list[str] = field(default_factory=list)


@dataclass
class FullPipelineCase:
    name: str
    objective: str


QUERY_REWRITE_CASES: list[QueryRewriteCase] = [
    QueryRewriteCase(
        name="acme_vs_globex_pricing",
        objective="Compare Acme and Globex pricing models",
        required_any_keywords=["pricing", "price", "acme", "globex", "cost"],
    ),
    QueryRewriteCase(
        name="market_size_growth",
        objective="What is the overall BI market size and growth rate?",
        required_any_keywords=["market size", "growth", "cagr", "market"],
    ),
]

SUPERVISOR_DECISION_CASES: list[SupervisorDecisionCase] = [
    SupervisorDecisionCase(
        name="research_done_analytics_pending",
        research_findings=[{"source": "x", "content": "y", "relevance_score": 0.9}],
        analytics_results=[],
        report_path=None,
        allowed_decisions=("research", "analytics"),
    ),
    SupervisorDecisionCase(
        name="research_and_analytics_done",
        research_findings=[{"source": "x", "content": "y", "relevance_score": 0.9}],
        analytics_results=[{"metric": "mean", "value": 1.0, "detail": "d"}],
        report_path=None,
        allowed_decisions=("research", "analytics", "reporting"),
    ),
]

ANALYTICS_CASES: list[AnalyticsCase] = [
    AnalyticsCase(
        name="globex_acv_range",
        objective="Compare Globex's minimum and maximum annual contract value",
        findings=[
            {
                "source": "competitor_globex.md",
                "content": (
                    "Industry estimates place typical annual contract value between "
                    "$150K and $400K depending on data volume."
                ),
                "relevance_score": 0.9,
            }
        ],
        min_tool_calls=1,
        # min, max, range, mean of 150000/400000 -- any of these appearing
        # means the model grounded its computation in the real figures.
        plausible_values=[150000.0, 400000.0, 250000.0, 275000.0],
        tolerance=1.0,
    ),
]

REPORTING_CASES: list[ReportingCase] = [
    ReportingCase(
        name="acme_pricing_report",
        objective="Summarize Acme's pricing strategy",
        findings=[
            {
                "source": "competitor_acme.md",
                "content": (
                    "Acme prices per seat with a data-volume overage component, "
                    "starting at $49 per seat per month on the Starter tier."
                ),
                "relevance_score": 0.9,
            }
        ],
        results=[{"metric": "starter_price", "value": 49.0, "detail": "starter tier price"}],
        required_facts=["49"],
    ),
]

FULL_PIPELINE_CASES: list[FullPipelineCase] = [
    FullPipelineCase(
        name="acme_vs_globex_full_run",
        objective="Assess Acme vs Globex pricing strategy and recommend a competitive positioning",
    ),
]
