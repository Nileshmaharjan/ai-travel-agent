import argparse
import uuid

from langgraph.types import Command

from travel_agents.config import get_settings
from travel_agents.graph import build_graph, plan_request
from travel_agents.llm import get_chat_model
from travel_agents.memory import open_checkpointer


def _stream(graph, payload, config) -> None:
    for update in graph.stream(payload, config, stream_mode="updates"):
        for node, data in update.items():
            if node == "supervisor":
                print(f"→ supervisor routes to: {data['next']}")
            elif node not in ("__interrupt__", "review") and data and data.get("messages"):
                print(f"✓ {node} reported")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a trip with a team of AI agents.")
    parser.add_argument("request", nargs="?", help='e.g. "5 days in Tokyo from Seoul in March, budget 1500 USD"')
    parser.add_argument("--thread", help="Conversation id. Reuse it to pick up a saved plan (needs DATABASE_URL).")
    args = parser.parse_args()

    thread = args.thread or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread}}

    with open_checkpointer(get_settings().database_url) as saver:
        graph = build_graph(get_chat_model(), checkpointer=saver)
        snapshot = graph.get_state(config)

        if snapshot.next:  # a saved run is paused at review
            print(f"Resuming thread {thread}")
        elif args.request:
            _stream(graph, plan_request(args.request), config)
        else:
            parser.error("give a trip request, or --thread for a plan awaiting review")

        while (snapshot := graph.get_state(config)).next:
            pending = snapshot.tasks[0].interrupts[0].value
            print("\n" + pending["plan"])
            print(f"\n[thread {thread}] revisions left: {pending['revisions_left']}")
            answer = input("Approve? Press Enter to approve, or type the changes you want: ").strip()
            decision = {"action": "approve"} if not answer else {"action": "revise", "feedback": answer}
            _stream(graph, Command(resume=decision), config)

        values = graph.get_state(config).values
        print("\nPlan approved." if values.get("approved") else "\nStopped without approval (revision limit reached).")


if __name__ == "__main__":
    main()
