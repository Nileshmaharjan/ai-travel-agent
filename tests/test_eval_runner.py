from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from travel_agents.evals.run import load_cases, run, to_markdown
from travel_agents.graph import build_graph
from tests.conftest import ScriptedChatModel, make_plan
from tests.test_graph import scripted_router

CASES_FILE = Path(__file__).parent.parent / "evals" / "cases.jsonl"


def test_case_file_is_valid_and_ids_unique():
    cases = load_cases(CASES_FILE)
    assert len(cases) >= 30
    assert len({c.id for c in cases}) == len(cases)
    assert all(c.end_date >= c.start_date for c in cases)
    assert any(c.expect_over_budget for c in cases)


def test_runner_scores_each_case_and_reports(monkeypatch):
    monkeypatch.setattr(
        "travel_agents.tools.search.post_json",
        lambda url, json, headers: {"results": [{"title": "H", "url": "https://example.com/hotel", "content": ""}]},
    )
    case = load_cases(CASES_FILE)[0]

    def factory():
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "hotels"}, "id": "c1"}]),
                AIMessage(content="Hotel report"),
            ]
        )
        days = make_plan().days  # wrong dates for this case on purpose
        planner = RunnableLambda(lambda _m: make_plan(start_date=case.start_date, end_date=case.end_date, days=days))
        return build_graph(model, router=scripted_router("lodging", "finish"), planner=planner, checkpointer=InMemorySaver())

    summary, results = run([case], factory)

    assert summary["cases"] == 1 and summary["errors"] == 0
    scores = {s["name"]: s["passed"] for s in results[0]["scores"]}
    assert scores == {"dates_valid": False, "budget_respected": True, "grounded_sources": True}
    assert "dates_valid" in to_markdown(summary, results)


def test_runner_records_errors_without_stopping():
    case = load_cases(CASES_FILE)[0]
    broken = RunnableLambda(lambda _m: (_ for _ in ()).throw(RuntimeError("model down")))
    summary, results = run(
        [case],
        lambda: build_graph(ScriptedChatModel(responses=[]), router=broken, checkpointer=InMemorySaver()),
    )
    assert summary["errors"] == 1
    assert "model down" in results[0]["error"]
