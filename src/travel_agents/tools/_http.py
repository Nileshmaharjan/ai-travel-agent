from typing import Any

import httpx

from travel_agents.config import get_settings


def get_json(url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    with httpx.Client(timeout=get_settings().http_timeout_s) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def post_json(url: str, *, json: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    with httpx.Client(timeout=get_settings().http_timeout_s) as client:
        resp = client.post(url, json=json, headers=headers)
        resp.raise_for_status()
        return resp.json()


def tool_error(exc: Exception) -> dict[str, str]:
    """Return failures to the model as data, so it can retry or explain instead of crashing the graph."""
    if isinstance(exc, httpx.HTTPStatusError):
        return {"error": f"upstream returned HTTP {exc.response.status_code}"}
    if isinstance(exc, httpx.TimeoutException):
        return {"error": "upstream timed out"}
    return {"error": f"{type(exc).__name__}: {exc}"}
