from datetime import date
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from travel_agents.tools._http import get_json, tool_error

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherInput(BaseModel):
    city: str = Field(description="City name, e.g. Kyoto")
    start_date: date = Field(description="First day, YYYY-MM-DD (forecasts reach ~16 days ahead)")
    end_date: date = Field(description="Last day, YYYY-MM-DD")


@tool("get_weather_forecast", args_schema=WeatherInput)
def get_weather_forecast(city: str, start_date: date, end_date: date) -> dict[str, Any]:
    """Daily min/max temperature (°C) and rain probability for a city over a date range."""
    if end_date < start_date:
        return {"error": "end_date is before start_date"}
    try:
        geo = get_json(GEOCODE_URL, params={"name": city, "count": 1})
        if not geo.get("results"):
            return {"error": f"unknown city: {city}"}
        place = geo["results"][0]
        fc = get_json(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return tool_error(exc)
    daily = fc.get("daily", {})
    days = [
        {"date": d, "max_c": hi, "min_c": lo, "rain_pct": rain}
        for d, hi, lo, rain in zip(
            daily.get("time", []),
            daily.get("temperature_2m_max", []),
            daily.get("temperature_2m_min", []),
            daily.get("precipitation_probability_max", []),
        )
    ]
    return {"city": place.get("name", city), "country": place.get("country"), "days": days}
