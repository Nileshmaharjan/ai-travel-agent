# AI Travel Agents

A multi-agent trip planner built with **LangGraph**. A supervisor routes a travel request to tool-calling
specialist agents (flights, lodging, itinerary) and a planner writes the final plan from their reports.

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
             ┌─────────┐
             │ planner │──► final plan
             └─────────┘
```

- **Supervisor:** the LLM returns a typed `Route` (structured output) naming the next specialist or `finish`.
  A step cap guarantees termination.
- **Specialists:** each is its own small graph (model → tool calls → model) where the model decides which
  tools to call and with what arguments. Tool-call rounds are capped per agent.
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
travel-agents "5 days in Tokyo from Seoul in early December, budget 1500 USD"
```

## Tests

```bash
pytest
```

The tests run offline. The HTTP tools are tested against mocked APIs (`respx`), and the graphs are driven by a
scripted chat model, so routing, the tool-call loop and the step caps are checked deterministically.

## Roadmap

- [x] **M1:** tool-calling specialists, supervisor routing, offline tests
- [ ] **M2:** Postgres checkpointer for conversation memory, human approval (`interrupt`) before anything is booked
- [ ] **M3:** evaluation suite: ~30 trip requests scored for budget compliance, date validity and grounding
      (every fact traceable to a tool result), run in CI
- [ ] **M4:** FastAPI service, Docker image, GitHub Actions, LLM tracing (LangSmith or Langfuse)
# ai-travel-agent
