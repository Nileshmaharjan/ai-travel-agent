import os
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from travel_agents.graph import build_graph, plan_request
from travel_agents.memory import open_checkpointer
from tests.conftest import ScriptedChatModel, scripted_planner
from tests.test_graph import scripted_router


def thread() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def test_run_pauses_for_review_then_approves():
    model = ScriptedChatModel(responses=[AIMessage(content="Flights report")])
    graph = build_graph(
        model, router=scripted_router("flights", "finish"), planner=scripted_planner("v1"), checkpointer=InMemorySaver()
    )
    config = thread()

    graph.invoke(plan_request("Tokyo trip"), config)
    snapshot = graph.get_state(config)
    assert snapshot.next == ("review",)
    pending = snapshot.tasks[0].interrupts[0].value
    assert pending["revisions_left"] == 3
    assert pending["plan"].startswith("# v1:")

    graph.invoke(Command(resume={"action": "approve"}), config)
    final = graph.get_state(config)
    assert final.next == ()
    assert final.values["approved"] is True


def test_revision_feedback_reaches_supervisor_and_replans():
    model = ScriptedChatModel(responses=[AIMessage(content="Cheaper hotels")])  # lodging specialist on revision
    graph = build_graph(
        model,
        router=scripted_router("finish", "lodging", "finish"),
        planner=scripted_planner("v1", "v2"),
        checkpointer=InMemorySaver(),
    )
    config = thread()

    graph.invoke(plan_request("Tokyo trip"), config)
    graph.invoke(Command(resume={"action": "revise", "feedback": "hotels under 100 USD"}), config)

    state = graph.get_state(config)
    feedback = [m for m in state.values["messages"] if isinstance(m, HumanMessage)][-1]
    assert feedback.content == "Please revise the plan: hotels under 100 USD"
    assert state.values["revisions"] == 1
    assert state.tasks[0].interrupts[0].value["plan"].startswith("# v2:")


def test_revision_limit_ends_without_approval():
    graph = build_graph(
        ScriptedChatModel(responses=[]),
        router=scripted_router("finish", "finish"),
        planner=scripted_planner("v1", "v2"),
        checkpointer=InMemorySaver(),
        max_revisions=1,
    )
    config = thread()

    graph.invoke(plan_request("Tokyo trip"), config)
    graph.invoke(Command(resume={"action": "revise", "feedback": "more museums"}), config)

    final = graph.get_state(config)
    assert final.next == ()
    assert final.values["approved"] is False


def test_malformed_resume_is_rejected():
    graph = build_graph(
        ScriptedChatModel(responses=[]), router=scripted_router("finish"), planner=scripted_planner("v1"),
        checkpointer=InMemorySaver(),
    )
    config = thread()
    graph.invoke(plan_request("Tokyo trip"), config)
    with pytest.raises(ValidationError):
        graph.invoke(Command(resume={"action": "book-it-now"}), config)


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to run against Postgres")
def test_paused_plan_survives_a_restart_in_postgres():
    url = os.environ["TEST_DATABASE_URL"]
    config = thread()

    with open_checkpointer(url) as saver:
        graph = build_graph(
            ScriptedChatModel(responses=[]), router=scripted_router("finish"), planner=scripted_planner("v1"),
            checkpointer=saver,
        )
        graph.invoke(plan_request("Tokyo"), config)

    # New connection and new graph object: only the database remembers the paused run.
    with open_checkpointer(url) as saver:
        graph = build_graph(ScriptedChatModel(responses=[]), router=scripted_router(), checkpointer=saver)
        assert graph.get_state(config).next == ("review",)
        graph.invoke(Command(resume={"action": "approve"}), config)
        assert graph.get_state(config).values["approved"] is True
