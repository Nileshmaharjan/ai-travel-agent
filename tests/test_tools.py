import httpx
import respx

from travel_agents.tools import convert_currency, get_weather_forecast, search_flights, web_search
from travel_agents.tools.currency import FRANKFURTER_URL
from travel_agents.tools.flights import AVIATIONSTACK_URL
from travel_agents.tools.search import TAVILY_URL
from travel_agents.tools.weather import FORECAST_URL, GEOCODE_URL


@respx.mock
def test_search_flights_normalises_response():
    route = respx.get(AVIATIONSTACK_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "airline": {"name": "Korean Air"},
                        "flight": {"iata": "KE703"},
                        "departure": {"scheduled": "2026-11-02T09:00:00+09:00"},
                        "arrival": {"scheduled": "2026-11-02T11:20:00+09:00"},
                    }
                ]
            },
        )
    )
    out = search_flights.invoke({"dep_iata": "icn", "arr_iata": "nrt"})
    assert route.calls.last.request.url.params["dep_iata"] == "ICN"
    assert out == {
        "route": "ICN-NRT",
        "flights": [
            {
                "airline": "Korean Air",
                "flight": "KE703",
                "departure": "2026-11-02T09:00:00+09:00",
                "arrival": "2026-11-02T11:20:00+09:00",
            }
        ],
    }


@respx.mock
def test_search_flights_returns_upstream_error_as_data():
    respx.get(AVIATIONSTACK_URL).mock(return_value=httpx.Response(500))
    assert search_flights.invoke({"dep_iata": "ICN", "arr_iata": "NRT"}) == {"error": "upstream returned HTTP 500"}


@respx.mock
def test_web_search_sends_bearer_key_and_trims_snippets():
    route = respx.post(TAVILY_URL).mock(
        return_value=httpx.Response(200, json={"results": [{"title": "Hotel", "url": "https://x", "content": "a" * 900}]})
    )
    out = web_search.invoke({"query": "hotels in Shinjuku"})
    assert route.calls.last.request.headers["Authorization"] == "Bearer tvly-test"
    assert len(out["results"][0]["snippet"]) == 500


@respx.mock
def test_weather_forecast_zips_daily_series():
    respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(200, json={"results": [{"name": "Kyoto", "country": "Japan", "latitude": 35.0, "longitude": 135.7}]})
    )
    respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "daily": {
                    "time": ["2026-11-02"],
                    "temperature_2m_max": [17.1],
                    "temperature_2m_min": [8.4],
                    "precipitation_probability_max": [60],
                }
            },
        )
    )
    out = get_weather_forecast.invoke({"city": "Kyoto", "start_date": "2026-11-02", "end_date": "2026-11-02"})
    assert out["days"] == [{"date": "2026-11-02", "max_c": 17.1, "min_c": 8.4, "rain_pct": 60}]


def test_weather_rejects_inverted_range():
    out = get_weather_forecast.invoke({"city": "Kyoto", "start_date": "2026-11-05", "end_date": "2026-11-02"})
    assert out == {"error": "end_date is before start_date"}


@respx.mock
def test_convert_currency():
    respx.get(FRANKFURTER_URL).mock(
        return_value=httpx.Response(200, json={"amount": 100.0, "base": "USD", "date": "2026-10-26", "rates": {"JPY": 15000.0}})
    )
    out = convert_currency.invoke({"amount": 100, "from_currency": "usd", "to_currency": "jpy"})
    assert out == {"amount": 15000.0, "currency": "JPY", "rate": 150.0, "as_of": "2026-10-26"}
