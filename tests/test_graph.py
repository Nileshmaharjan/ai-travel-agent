from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from travel_agents.agents import SPECIALISTS, build_specialist
from travel_agents.graph import Route, build_graph, plan_request
from tests.conftest import ScriptedChatModel, scripted_planner


def scripted_router(*decisions: str) -> RunnableLambda:
    queue = list(decisions)
    return RunnableLambda(lambda _messages: Route(next=queue.pop(0), reason="test"))


def test_specialist_runs_tool_loop(monkeypatch):
    monkeypatch.setattr(
        "travel_agents.tools.flights.get_json",
        lambda url, params: {"data": [{"airline": {"name": "Asiana"}, "flight": {"iata": "OZ102"}}]},
    )
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "search_flights", "args": {"dep_iata": "ICN", "arr_iata": "NRT"}, "id": "c1"}]),
            AIMessage(content="Asiana OZ102 flies ICN to NRT."),
        ]
    )
    flights = next(s for s in SPECIALISTS if s.name == "flights")
    result = build_specialist(flights, model).invoke({"messages": [("user", "Seoul to Tokyo")]})

    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "OZ102" in tool_msg.content
    assert result["messages"][-1].content == "Asiana OZ102 flies ICN to NRT."


def test_specialist_stops_after_max_tool_rounds():
    looping = AIMessage(content="", tool_calls=[{"name": "convert_currency", "args": {"amount": 1, "from_currency": "USD", "to_currency": "USD"}, "id": "c"}])
    model = ScriptedChatModel(responses=[looping.model_copy() for _ in range(3)])
    flights = next(s for s in SPECIALISTS if s.name == "flights")
    result = build_specialist(flights, model, max_tool_rounds=1).invoke({"messages": [("user", "x")]})
    assert sum(isinstance(m, ToolMessage) for m in result["messages"]) == 1


def test_supervisor_routes_specialists_then_plans():
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="Flights: KE703 ICN-NRT."),   # flights specialist (no tool call)
            AIMessage(content="Hotels: 3 options."),         # lodging specialist
        ]
    )
    graph = build_graph(model, router=scripted_router("flights", "lodging", "finish"), planner=scripted_planner("FINAL"))
    out = graph.invoke(plan_request("3 days in Tokyo from Seoul"))

    names = [m.name for m in out["messages"] if isinstance(m, AIMessage) and m.name]
    assert names == ["flights", "lodging", "planner"]
    assert out["plan"]["destination"] == "FINAL"
    assert out["messages"][-1].content.startswith("# FINAL: 2026-11-05 to 2026-11-06")


def test_supervisor_step_cap_forces_finish():
    model = ScriptedChatModel(responses=[AIMessage(content="report")])
    graph = build_graph(
        model, router=scripted_router("flights", "flights", "flights"), planner=scripted_planner("FINAL"), max_steps=1
    )
    out = graph.invoke(plan_request("x"))
    assert out["steps"] == 1
    assert out["plan"]["destination"] == "FINAL"


def test_specialist_tool_urls_are_recorded_as_evidence(monkeypatch):
    monkeypatch.setattr(
        "travel_agents.tools.search.post_json",
        lambda url, json, headers: {"results": [{"title": "Hotel A", "url": "https://hotels.example/a", "content": "x"}]},
    )
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "hotels"}, "id": "c1"}]),
            AIMessage(content="Hotel A looks good (https://made-up.example)."),
        ]
    )
    graph = build_graph(model, router=scripted_router("lodging", "finish"), planner=scripted_planner("T"))
    out = graph.invoke(plan_request("x"))
    # Only what the tool returned counts; a URL the model wrote in its report does not.
    assert out["evidence"] == ["https://hotels.example/a"]
