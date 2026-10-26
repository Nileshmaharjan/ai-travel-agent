from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from travel_agents.config import get_settings
from travel_agents.tools._http import post_json, tool_error

TAVILY_URL = "https://api.tavily.com/search"


class WebSearchInput(BaseModel):
    query: str = Field(description="What to search for, e.g. 'mid-range hotels near Shinjuku station'")
    max_results: int = Field(default=5, ge=1, le=10)


@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web for hotels, attractions, opening hours and prices. Returns titles, URLs and snippets."""
    key = get_settings().tavily_api_key
    if not key:
        return {"error": "TAVILY_API_KEY is not set"}
    try:
        data = post_json(
            TAVILY_URL,
            json={"query": query, "max_results": max_results},
            headers={"Authorization": f"Bearer {key}"},
        )
    except Exception as exc:  # noqa: BLE001
        return tool_error(exc)
    results = [
        {"title": r.get("title"), "url": r.get("url"), "snippet": (r.get("content") or "")[:500]}
        for r in data.get("results", [])
    ]
    return {"query": query, "results": results}
