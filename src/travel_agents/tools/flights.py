from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from travel_agents.config import get_settings
from travel_agents.tools._http import get_json, tool_error

# The AviationStack free plan is HTTP-only.
AVIATIONSTACK_URL = "http://api.aviationstack.com/v1/flights"


class FlightSearchInput(BaseModel):
    dep_iata: str = Field(description="Departure airport IATA code, e.g. ICN", min_length=3, max_length=3)
    arr_iata: str = Field(description="Arrival airport IATA code, e.g. NRT", min_length=3, max_length=3)
    limit: int = Field(default=5, ge=1, le=10, description="Maximum number of flights to return")


@tool("search_flights", args_schema=FlightSearchInput)
def search_flights(dep_iata: str, arr_iata: str, limit: int = 5) -> dict[str, Any]:
    """Find scheduled flights between two airports. Returns airline, flight number and scheduled times."""
    key = get_settings().aviationstack_api_key
    if not key:
        return {"error": "AVIATIONSTACK_API_KEY is not set"}
    try:
        data = get_json(
            AVIATIONSTACK_URL,
            params={"access_key": key, "dep_iata": dep_iata.upper(), "arr_iata": arr_iata.upper(), "limit": limit},
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as data
        return tool_error(exc)
    if "error" in data:
        return {"error": data["error"].get("message", "AviationStack error")}
    flights = [
        {
            "airline": (f.get("airline") or {}).get("name"),
            "flight": (f.get("flight") or {}).get("iata"),
            "departure": (f.get("departure") or {}).get("scheduled"),
            "arrival": (f.get("arrival") or {}).get("scheduled"),
        }
        for f in data.get("data", [])[:limit]
    ]
    return {"route": f"{dep_iata.upper()}-{arr_iata.upper()}", "flights": flights}
