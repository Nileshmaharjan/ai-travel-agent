"""Deterministic scorers. Each takes an eval case, the typed plan and the tool evidence, and returns a Score.

No LLM-as-judge here on purpose: these checks are cheap, reproducible and cannot be talked out of a failure.
"""

from collections.abc import Callable
from datetime import date, timedelta

from pydantic import BaseModel

from travel_agents.plan import Money, TripPlan


class EvalCase(BaseModel):
    id: str
    request: str
    destination: str
    start_date: date
    end_date: date
    budget: Money | None = None
    expect_over_budget: bool = False  # budget deliberately too small: the plan must say so


class Score(BaseModel):
    name: str
    passed: bool
    detail: str = ""


def dates_valid(case: EvalCase, plan: TripPlan, evidence: list[str]) -> Score:
    """The plan covers exactly the requested days, once each, in order."""
    expected = [case.start_date + timedelta(days=i) for i in range((case.end_date - case.start_date).days + 1)]
    got = [d.date for d in plan.days]
    if (plan.start_date, plan.end_date) != (case.start_date, case.end_date):
        return Score(name="dates_valid", passed=False, detail=f"range {plan.start_date}..{plan.end_date} != requested")
    if got != expected:
        return Score(name="dates_valid", passed=False, detail=f"{len(got)} day entries, expected {len(expected)} consecutive")
    return Score(name="dates_valid", passed=True)


def budget_respected(case: EvalCase, plan: TripPlan, evidence: list[str]) -> Score:
    """Within budget, in the budget's currency; or, for impossible budgets, honestly flagged as over."""
    name = "budget_respected"
    if case.budget is None:
        return Score(name=name, passed=True, detail="no budget given")
    total = plan.estimated_total
    if total is None:
        ok = bool(plan.gaps)
        return Score(name=name, passed=ok, detail="no total; gap declared" if ok else "no total and no gap declared")
    if total.currency.upper() != case.budget.currency.upper():
        return Score(name=name, passed=False, detail=f"total in {total.currency}, budget in {case.budget.currency}")
    over = total.amount > case.budget.amount
    if over != plan.over_budget:
        return Score(name=name, passed=False, detail=f"over_budget flag is {plan.over_budget} but total {total.amount} vs {case.budget.amount}")
    if case.expect_over_budget:
        return Score(name=name, passed=over, detail="impossible budget flagged" if over else "claims an impossible budget fits")
    return Score(name=name, passed=not over, detail=f"{total.amount:.0f} of {case.budget.amount:.0f} {total.currency}")


def grounded_sources(case: EvalCase, plan: TripPlan, evidence: list[str]) -> Score:
    """Every cited URL was returned by a tool during this run, and a non-empty plan cites something."""
    name = "grounded_sources"
    seen = {u.rstrip("/.,") for u in evidence}
    invented = [s for s in plan.sources if s.rstrip("/.,") not in seen]
    if invented:
        return Score(name=name, passed=False, detail=f"{len(invented)} source(s) not returned by any tool: {invented[:3]}")
    if plan.days and not plan.sources:
        return Score(name=name, passed=False, detail="plan has days but cites no sources")
    return Score(name=name, passed=True, detail=f"{len(plan.sources)} source(s) verified")


SCORERS: tuple[Callable[[EvalCase, TripPlan, list[str]], Score], ...] = (
    dates_valid,
    budget_respected,
    grounded_sources,
)


def score_plan(case: EvalCase, plan: TripPlan, evidence: list[str]) -> list[Score]:
    return [scorer(case, plan, evidence) for scorer in SCORERS]
