# AI Travel Agents

A multi-agent trip planner built with **LangGraph**. A supervisor routes a travel request to tool-calling
specialist agents (flights, lodging, itinerary), a planner writes the final plan from their reports, and the
traveller approves it or asks for changes. Runs pause and resume from Postgres, so a plan can wait for review
across restarts.

```
                 ┌──────────────┐
  request ──────►│  supervisor  │◄─────────────────────┐
                 └──────┬───────┘                      │
          structured    │ Route(next=…)                │ report
          output        ▼                              │
   ┌───────────┬────────────────┬───────────────┐      │
   │  flights  │    lodging     │   itinerary   │──────┘
   │ agent ⇄   │  agent ⇄ tools │ agent ⇄ tools │
   │ tools     │                │               │
   └───────────┴────────────────┴───────────────┘
                 │ next = finish (or step cap)
                 ▼
             ┌─────────┐      ┌──────────────────────┐
             │ planner │─────►│ review (interrupt)   │──► approve ──► done
             └─────────┘      └──────────┬───────────┘
                                         │ revise + feedback
                                         └──────────► supervisor
```

- **Supervisor:** the LLM returns a typed `Route` (structured output) naming the next specialist or `finish`.
  A step cap guarantees termination.
- **Specialists:** each is its own small graph (model → tool calls → model) where the model decides which
  tools to call and with what arguments. Tool-call rounds are capped per agent.
- **Human-in-the-loop review:** `interrupt()` pauses the run with the draft plan. The traveller resumes it with
  `Command(resume={"action": "approve"})` or `{"action": "revise", "feedback": …}`. The payload is validated, and
  revisions are capped.
- **Memory:** a Postgres checkpointer stores every step under a `thread_id`, so a paused plan survives restarts
  and can be picked up later with `--thread`. Without `DATABASE_URL` it falls back to in-memory checkpoints.
- **Typed plan:** the planner returns a `TripPlan` (structured output): one entry per day, an estimated total in
  the budget currency, an `over_budget` flag, sources and declared gaps. The graph also records every URL the
  tools actually returned (`evidence`), which is what grounding is checked against.
- **Tools:** typed Pydantic input schemas; failures come back to the model as data (`{"error": …}`) so it
  can recover or report the gap instead of inventing facts.

| Tool | Source | Key needed |
|------|--------|------------|
| `search_flights` | AviationStack | yes |
| `web_search` | Tavily | yes |
| `get_weather_forecast` | Open-Meteo | no |
| `convert_currency` | Frankfurter (ECB rates) | no |

The default model is Llama 3.3 70B on Groq. Every agent takes a LangChain `BaseChatModel`, so the model can be swapped.

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add your keys; .env is git-ignored
docker compose up -d        # Postgres on localhost:5433 (optional; omit DATABASE_URL for in-memory)
travel-agents "5 days in Tokyo from Seoul in early December, budget 1500 USD"
```

The run stops at review: press Enter to approve or type the changes you want. Note the printed thread id and
run `travel-agents --thread <id>` later to pick up a plan that is still awaiting review.

## Tests

```bash
pytest
```

The tests run offline. Set `TEST_DATABASE_URL=postgresql://travel:travel@localhost:5433/travel_agents` to also run
the Postgres test, which pauses a plan, reconnects and resumes it. The HTTP tools are tested against mocked APIs (`respx`), and the graphs are driven by a
scripted chat model, so routing, the tool-call loop and the step caps are checked deterministically.

## Evaluation

`evals/cases.jsonl` holds 30 trip requests across Asia-Pacific, Europe and the US, in 11 currencies. It includes
deliberately impossible budgets, where the correct answer is to report that the plan is over budget. Each plan is
scored by deterministic checks, with no LLM judge:

| Scorer | Passes when |
|---|---|
| `dates_valid` | the plan covers exactly the requested dates, one entry per day, in order |
| `budget_respected` | the total is in the budget currency and within budget, the `over_budget` flag matches the numbers, and impossible budgets are flagged rather than claimed to fit |
| `grounded_sources` | every cited URL was returned by a tool during the run (none invented), and a non-empty plan cites at least one |

```bash
travel-agents-eval --limit 5            # live run; writes evals/results/<timestamp>.{json,md}
```

CI (`.github/workflows/ci.yml`) runs the offline tests and the Postgres test on every push. `evals.yml` runs the
live eval weekly or on demand, using API keys stored as repository secrets, and posts the score table to the job
summary.

## Roadmap

- [x] **M1:** tool-calling specialists, supervisor routing, offline tests
- [x] **M2:** Postgres checkpointer for conversation memory, human review (`interrupt`) before a plan is final
- [x] **M3:** evaluation suite: 30 trip requests scored for date validity, budget compliance and grounding;
      tests in CI on every push, live evals on a schedule
- [ ] **M4:** FastAPI service, Docker image, GitHub Actions, LLM tracing (LangSmith or Langfuse)
# ai-travel-agent
