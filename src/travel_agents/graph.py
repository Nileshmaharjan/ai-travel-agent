"""Supervisor graph: a router picks the next specialist until the plan is ready, then a planner writes it up."""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from travel_agents.agents import SPECIALISTS, build_specialist

AgentName = Literal["flights", "lodging", "itinerary", "finish"]


class Route(BaseModel):
    """The supervisor's decision about who acts next."""

    next: AgentName = Field(description="The specialist to call next, or 'finish' when the plan can be written")
    reason: str = Field(description="One sentence on why")


class TravelState(MessagesState):
    next: str
    steps: int


SUPERVISOR_PROMPT = """You coordinate a team of travel specialists:
{team}

Read the conversation, including each specialist's report so far, and choose who acts next.
Call each specialist only when the request needs it and it has not already reported.
Choose "finish" once every part of the request has been covered."""

PLANNER_PROMPT = """You write the final trip plan from the specialists' reports in the conversation.
Use only facts from those reports and keep their source URLs. Check the total against the traveller's budget
and say clearly if it is over. Where a report says data was unavailable, say so instead of filling the gap."""


def build_graph(
    model: BaseChatModel,
    *,
    router: Runnable | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    max_steps: int = 6,
) -> CompiledStateGraph:
    """Build the planner. `router` defaults to `model` with structured output; tests inject a scripted one."""
    team = "\n".join(f"- {s.name}: {s.description}" for s in SPECIALISTS)
    router = router or model.with_structured_output(Route)

    def supervisor(state: TravelState) -> dict:
        steps = state.get("steps", 0)
        if steps >= max_steps:
            return {"next": "finish", "steps": steps}
        decision: Route = router.invoke([SystemMessage(SUPERVISOR_PROMPT.format(team=team)), *state["messages"]])
        return {"next": decision.next, "steps": steps + 1}

    def make_specialist_node(spec):
        agent = build_specialist(spec, model)

        def run(state: TravelState) -> dict:
            # Specialists see the request and earlier reports, not each other's raw tool traffic.
            result = agent.invoke({"messages": state["messages"]})
            report = result["messages"][-1].content
            return {"messages": [AIMessage(content=report, name=spec.name)]}

        return run

    def planner(state: TravelState) -> dict:
        return {"messages": [model.invoke([SystemMessage(PLANNER_PROMPT), *state["messages"]])]}

    graph = StateGraph(TravelState)
    graph.add_node("supervisor", supervisor)
    for spec in SPECIALISTS:
        graph.add_node(spec.name, make_specialist_node(spec))
        graph.add_edge(spec.name, "supervisor")
    graph.add_node("planner", planner)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda s: "planner" if s["next"] == "finish" else s["next"],
        [*(s.name for s in SPECIALISTS), "planner"],
    )
    graph.add_edge("planner", END)
    return graph.compile(checkpointer=checkpointer)


def plan_request(text: str) -> dict:
    return {"messages": [HumanMessage(content=text)], "steps": 0}
