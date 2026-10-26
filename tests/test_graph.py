from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from travel_agents.agents import SPECIALISTS, build_specialist
from travel_agents.graph import Route, build_graph, plan_request
from tests.conftest import ScriptedChatModel


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
            AIMessage(content="FINAL PLAN"),                 # planner
        ]
    )
    graph = build_graph(model, router=scripted_router("flights", "lodging", "finish"))
    out = graph.invoke(plan_request("3 days in Tokyo from Seoul"))

    names = [m.name for m in out["messages"] if isinstance(m, AIMessage) and m.name]
    assert names == ["flights", "lodging"]
    assert out["messages"][-1].content == "FINAL PLAN"


def test_supervisor_step_cap_forces_finish():
    model = ScriptedChatModel(responses=[AIMessage(content="report"), AIMessage(content="FINAL PLAN")])
    graph = build_graph(model, router=scripted_router("flights", "flights", "flights"), max_steps=1)
    out = graph.invoke(plan_request("x"))
    assert out["steps"] == 1
    assert out["messages"][-1].content == "FINAL PLAN"
