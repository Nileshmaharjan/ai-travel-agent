"""Specialist agents. Each one is a small LangGraph loop: model -> tool calls -> model, until it answers."""

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from travel_agents.tools import convert_currency, get_weather_forecast, search_flights, web_search


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str  # what the supervisor reads when choosing who acts next
    system_prompt: str
    tools: tuple[BaseTool, ...]


SPECIALISTS: tuple[AgentSpec, ...] = (
    AgentSpec(
        name="flights",
        description="Finds flights between airports and reports airlines and scheduled times.",
        system_prompt=(
            "You are the flight specialist. Work out the departure and arrival airports (IATA codes) from the "
            "request and call search_flights. Report only what the tool returned; if it returns an error or no "
            "flights, say so plainly. Never invent flight numbers, times or prices."
        ),
        tools=(search_flights, convert_currency),
    ),
    AgentSpec(
        name="lodging",
        description="Researches hotels and areas to stay that fit the budget.",
        system_prompt=(
            "You are the lodging specialist. Use web_search to find 3 options that fit the traveller's budget and "
            "area preferences. Use convert_currency when prices and budget are in different currencies. "
            "Include each option's source URL. Never invent prices."
        ),
        tools=(web_search, convert_currency),
    ),
    AgentSpec(
        name="itinerary",
        description="Builds a day-by-day itinerary using attractions research and the weather forecast.",
        system_prompt=(
            "You are the itinerary specialist. Use web_search for attractions and get_weather_forecast for the "
            "trip dates when they are within the forecast window, and plan indoor options on rainy days. "
            "Produce a day-by-day plan with source URLs."
        ),
        tools=(web_search, get_weather_forecast),
    ),
)


def build_specialist(
    spec: AgentSpec, model: BaseChatModel, *, max_tool_rounds: int = 4
) -> CompiledStateGraph:
    """Compile one tool-calling agent. The model chooses which tools to call and with what arguments."""
    tools: Sequence[BaseTool] = spec.tools
    bound = model.bind_tools(list(tools))

    def call_model(state: MessagesState) -> dict:
        return {"messages": [bound.invoke([SystemMessage(spec.system_prompt), *state["messages"]])]}

    def route(state: MessagesState) -> str:
        last = state["messages"][-1]
        rounds = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)
        if isinstance(last, AIMessage) and last.tool_calls and rounds <= max_tool_rounds:
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(list(tools)))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, ["tools", END])
    graph.add_edge("tools", "model")
    return graph.compile(name=spec.name)
