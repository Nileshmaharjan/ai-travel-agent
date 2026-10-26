from datetime import date

from travel_agents.evals.scorers import EvalCase, budget_respected, dates_valid, grounded_sources
from travel_agents.plan import DayPlan, Money
from tests.conftest import make_plan

CASE = EvalCase(
    id="t",
    request="2 days in Tokyo",
    destination="Tokyo",
    start_date=date(2026, 11, 5),
    end_date=date(2026, 11, 6),
    budget=Money(amount=1500, currency="USD"),
)
EVIDENCE = ["https://example.com/hotel"]


def test_dates_valid_passes_for_exact_consecutive_days():
    assert dates_valid(CASE, make_plan(), EVIDENCE).passed


def test_dates_valid_fails_on_missing_day():
    plan = make_plan(days=[DayPlan(date=date(2026, 11, 5), title="Only day", activities=[])])
    assert not dates_valid(CASE, plan, EVIDENCE).passed


def test_dates_valid_fails_on_wrong_range():
    assert not dates_valid(CASE, make_plan(end_date=date(2026, 11, 7)), EVIDENCE).passed


def test_budget_passes_within_budget():
    assert budget_respected(CASE, make_plan(), EVIDENCE).passed


def test_budget_fails_when_over():
    plan = make_plan(estimated_total=Money(amount=2000, currency="USD"), over_budget=True)
    assert not budget_respected(CASE, plan, EVIDENCE).passed


def test_budget_fails_on_dishonest_flag():
    plan = make_plan(estimated_total=Money(amount=2000, currency="USD"), over_budget=False)
    score = budget_respected(CASE, plan, EVIDENCE)
    assert not score.passed and "flag" in score.detail


def test_budget_fails_on_currency_mismatch():
    plan = make_plan(estimated_total=Money(amount=900, currency="EUR"))
    assert not budget_respected(CASE, plan, EVIDENCE).passed


def test_budget_without_total_needs_declared_gap():
    assert not budget_respected(CASE, make_plan(estimated_total=None), EVIDENCE).passed
    assert budget_respected(CASE, make_plan(estimated_total=None, gaps=["hotel prices"]), EVIDENCE).passed


def test_impossible_budget_must_be_flagged():
    case = CASE.model_copy(update={"budget": Money(amount=100, currency="USD"), "expect_over_budget": True})
    honest = make_plan(estimated_total=Money(amount=900, currency="USD"), budget=case.budget, over_budget=True)
    assert budget_respected(case, honest, EVIDENCE).passed


def test_grounded_sources_rejects_invented_urls():
    plan = make_plan(sources=["https://example.com/hotel", "https://invented.example/x"])
    score = grounded_sources(CASE, plan, EVIDENCE)
    assert not score.passed and "invented.example" in score.detail


def test_grounded_sources_requires_a_citation():
    assert not grounded_sources(CASE, make_plan(sources=[]), EVIDENCE).passed


def test_grounded_sources_ignores_trailing_slash():
    assert grounded_sources(CASE, make_plan(sources=["https://example.com/hotel/"]), EVIDENCE).passed
