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

## Roadmap

- [x] **M1:** tool-calling specialists, supervisor routing, offline tests
- [x] **M2:** Postgres checkpointer for conversation memory, human review (`interrupt`) before a plan is final
- [ ] **M3:** evaluation suite: ~30 trip requests scored for budget compliance, date validity and grounding
      (every fact traceable to a tool result), run in CI
- [ ] **M4:** FastAPI service, Docker image, GitHub Actions, LLM tracing (LangSmith or Langfuse)
# ai-travel-agent
