"""Run the eval cases through the graph and write a JSON + Markdown report.

    travel-agents-eval --cases evals/cases.jsonl --limit 5
"""

import argparse
import json
import sys
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from travel_agents.evals.scorers import SCORERS, EvalCase, score_plan
from travel_agents.graph import build_graph, plan_request
from travel_agents.plan import TripPlan


def load_cases(path: Path) -> list[EvalCase]:
    return [EvalCase.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def run_case(graph: CompiledStateGraph, case: EvalCase) -> dict:
    config = {"configurable": {"thread_id": f"eval-{case.id}-{uuid.uuid4().hex[:6]}"}}
    started = time.perf_counter()
    try:
        graph.invoke(plan_request(case.request), config)  # stops at the review interrupt
        values = graph.get_state(config).values
        plan = TripPlan.model_validate(values["plan"])
        scores = score_plan(case, plan, values.get("evidence", []))
        error = None
    except Exception as exc:  # noqa: BLE001 - one broken case must not stop the run
        scores, error = [], f"{type(exc).__name__}: {exc}"
    return {
        "id": case.id,
        "passed": error is None and all(s.passed for s in scores),
        "error": error,
        "seconds": round(time.perf_counter() - started, 1),
        "scores": [s.model_dump() for s in scores],
    }


def summarise(results: list[dict]) -> dict:
    by_scorer = {}
    for scorer in SCORERS:
        name = scorer.__name__
        vals = [s["passed"] for r in results for s in r["scores"] if s["name"] == name]
        by_scorer[name] = round(sum(vals) / len(vals), 3) if vals else None
    return {
        "cases": len(results),
        "pass_rate": round(sum(r["passed"] for r in results) / len(results), 3) if results else 0.0,
        "errors": sum(r["error"] is not None for r in results),
        "by_scorer": by_scorer,
    }


def to_markdown(summary: dict, results: list[dict]) -> str:
    lines = [
        f"## Eval results: {summary['pass_rate']:.0%} of {summary['cases']} cases passed",
        "",
        "| Scorer | Pass rate |",
        "|---|---|",
        *(f"| {k} | {'n/a' if v is None else f'{v:.0%}'} |" for k, v in summary["by_scorer"].items()),
        "",
        "| Case | Result | Details |",
        "|---|---|---|",
    ]
    for r in results:
        detail = r["error"] or "; ".join(f"{s['name']}: {s['detail']}" for s in r["scores"] if not s["passed"])
        lines.append(f"| {r['id']} | {'✅' if r['passed'] else '❌'} | {detail or ''} |")
    return "\n".join(lines)


def run(cases: list[EvalCase], graph_factory: Callable[[], CompiledStateGraph]) -> tuple[dict, list[dict]]:
    graph = graph_factory()
    results = [run_case(graph, case) for case in cases]
    return summarise(results), results


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the travel planner against evals/cases.jsonl")
    parser.add_argument("--cases", type=Path, default=Path("evals/cases.jsonl"))
    parser.add_argument("--limit", type=int, default=0, help="run only the first N cases (0 = all)")
    parser.add_argument("--out", type=Path, default=Path("evals/results"))
    parser.add_argument("--min-pass-rate", type=float, default=0.0, help="exit 1 below this overall pass rate")
    args = parser.parse_args()

    from travel_agents.llm import get_chat_model

    cases = load_cases(args.cases)[: args.limit or None]
    summary, results = run(cases, lambda: build_graph(get_chat_model(), checkpointer=InMemorySaver()))

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (args.out / f"{stamp}.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    report = to_markdown(summary, results)
    (args.out / f"{stamp}.md").write_text(report + "\n")
    print(report)
    sys.exit(1 if summary["pass_rate"] < args.min_pass_rate else 0)


if __name__ == "__main__":
    main()
