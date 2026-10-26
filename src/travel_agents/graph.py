"""Supervisor graph: a router picks the next specialist until the plan is ready, a planner writes it up,
and the traveller approves it or asks for changes (human-in-the-loop via `interrupt`)."""

import operator
import re
from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from travel_agents.agents import SPECIALISTS, build_specialist
from travel_agents.plan import TripPlan

AgentName = Literal["flights", "lodging", "itinerary", "finish"]
URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")


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
    plan: dict | None
    # URLs that tools actually returned. The grounding eval checks the plan's sources against this.
    evidence: Annotated[list[str], operator.add]


SUPERVISOR_PROMPT = """You coordinate a team of travel specialists:
{team}

Read the conversation, including each specialist's report so far, and choose who acts next.
Call each specialist only when the request needs it and it has not already reported.
Choose "finish" once every part of the request has been covered."""

PLANNER_PROMPT = """You write the final trip plan from the specialists' reports in the conversation.
- Use only facts from those reports. Every URL in `sources` must appear in a report.
- Include one day entry per calendar day from start_date to end_date.
- Express estimated_total in the traveller's budget currency, using converted figures from the reports.
  If prices were unavailable, set estimated_total to null and list what is missing in `gaps`.
- Set over_budget honestly. Where a report says data was unavailable, add it to `gaps` instead of guessing."""


def build_graph(
    model: BaseChatModel,
    *,
    router: Runnable | None = None,
    planner: Runnable | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    max_steps: int = 6,
    max_revisions: int = 3,
) -> CompiledStateGraph:
    """Build the planner graph. `router` and `planner` default to `model` with structured output
    (Route and TripPlan); tests inject scripted ones."""
    team = "\n".join(f"- {s.name}: {s.description}" for s in SPECIALISTS)
    router = router or model.with_structured_output(Route)
    planner = planner or model.with_structured_output(TripPlan)

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
            urls = [
                url
                for m in result["messages"]
                if isinstance(m, ToolMessage)
                for url in URL_RE.findall(str(m.content))
            ]
            return {"messages": [AIMessage(content=report, name=spec.name)], "evidence": urls}

        return run

    def write_plan(state: TravelState) -> dict:
        plan: TripPlan = planner.invoke([SystemMessage(PLANNER_PROMPT), *state["messages"]])
        return {"plan": plan.model_dump(mode="json"), "messages": [AIMessage(content=plan.to_markdown(), name="planner")]}

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
    graph.add_node("planner", write_plan)
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
    return {
        "messages": [HumanMessage(content=text)],
        "steps": 0,
        "revisions": 0,
        "approved": False,
        "plan": None,
        "evidence": [],
    }
