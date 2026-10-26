from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from travel_agents.config import Settings, get_settings


class ScriptedChatModel(BaseChatModel):
    """Returns pre-written AIMessages in order, so graph tests run offline and deterministically."""

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("ScriptedChatModel ran out of responses")
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        return self


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    settings = Settings(groq_api_key="test", tavily_api_key="tvly-test", aviationstack_api_key="av-test")
    get_settings.cache_clear()
    monkeypatch.setattr("travel_agents.config.get_settings", lambda: settings)
    for mod in ("flights", "search"):
        monkeypatch.setattr(f"travel_agents.tools.{mod}.get_settings", lambda: settings)
    monkeypatch.setattr("travel_agents.tools._http.get_settings", lambda: settings)
    return settings


def make_plan(destination: str = "Tokyo", **overrides: Any) -> "TripPlan":
    from datetime import date

    from travel_agents.plan import DayPlan, Money, TripPlan

    fields = dict(
        destination=destination,
        start_date=date(2026, 11, 5),
        end_date=date(2026, 11, 6),
        days=[DayPlan(date=date(2026, 11, 5), title="Arrive", activities=["Shinjuku"]),
              DayPlan(date=date(2026, 11, 6), title="Leave", activities=["Asakusa"])],
        estimated_total=Money(amount=900, currency="USD"),
        budget=Money(amount=1500, currency="USD"),
        over_budget=False,
        sources=["https://example.com/hotel"],
        gaps=[],
    )
    fields.update(overrides)
    return TripPlan(**fields)


def scripted_planner(*destinations: str):
    from langchain_core.runnables import RunnableLambda

    queue = list(destinations)
    return RunnableLambda(lambda _messages: make_plan(queue.pop(0)))
