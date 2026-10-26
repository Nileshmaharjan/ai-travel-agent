"""Supervisor graph: a router picks the next specialist until the plan is ready, a planner writes it up,
and the traveller approves it or asks for changes (human-in-the-loop via `interrupt`)."""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from travel_agents.agents import SPECIALISTS, build_specialist

AgentName = Literal["flights", "lodging", "itinerary", "finish"]


class Route(BaseModel):
    """The supervisor's decision about who acts next."""

    next: AgentName = Field(description="The specialist to call next, or 'finish' when the plan can be written")
    reason: str = Field(description="One sentence on why")


class ReviewDecision(BaseModel):
    """What the traveller sends back when the graph pauses for review."""

    action: Literal["approve", "revise"]
    feedback: str = ""


class TravelState(MessagesState):
    next: str
    steps: int
    revisions: int
    approved: bool


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
    max_revisions: int = 3,
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

    def review(state: TravelState) -> Command:
        revisions = state.get("revisions", 0)
        if revisions >= max_revisions:
            return Command(goto=END, update={"approved": False})
        # Pauses the run and persists state in the checkpointer; resumed with Command(resume={...}).
        raw = interrupt({"plan": state["messages"][-1].content, "revisions_left": max_revisions - revisions})
        decision = ReviewDecision.model_validate(raw)
        if decision.action == "approve":
            return Command(goto=END, update={"approved": True})
        return Command(
            goto="supervisor",
            update={
                "messages": [HumanMessage(content=f"Please revise the plan: {decision.feedback}")],
                "steps": 0,
                "revisions": revisions + 1,
            },
        )

    graph = StateGraph(TravelState)
    graph.add_node("supervisor", supervisor)
    for spec in SPECIALISTS:
        graph.add_node(spec.name, make_specialist_node(spec))
        graph.add_edge(spec.name, "supervisor")
    graph.add_node("planner", planner)
    graph.add_node("review", review, destinations=("supervisor", END))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda s: "planner" if s["next"] == "finish" else s["next"],
        [*(s.name for s in SPECIALISTS), "planner"],
    )
    graph.add_edge("planner", "review")
    return graph.compile(checkpointer=checkpointer)


def plan_request(text: str) -> dict:
    return {"messages": [HumanMessage(content=text)], "steps": 0, "revisions": 0, "approved": False}
