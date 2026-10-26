import argparse
import uuid

from langgraph.checkpoint.memory import InMemorySaver

from travel_agents.graph import build_graph, plan_request
from travel_agents.llm import get_chat_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a trip with a team of AI agents.")
    parser.add_argument("request", help='e.g. "5 days in Tokyo from Seoul in March, budget 1500 USD"')
    args = parser.parse_args()

    graph = build_graph(get_chat_model(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    final = None
    for update in graph.stream(plan_request(args.request), config, stream_mode="updates"):
        for node, payload in update.items():
            if node == "supervisor":
                print(f"→ supervisor routes to: {payload['next']}")
            elif payload and payload.get("messages"):
                final = payload["messages"][-1].content
                print(f"✓ {node} reported")
    print("\n" + (final or "No plan produced."))


if __name__ == "__main__":
    main()
