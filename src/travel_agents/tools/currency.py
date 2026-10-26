from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from travel_agents.tools._http import get_json, tool_error

FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest"


class CurrencyInput(BaseModel):
    amount: float = Field(gt=0, description="Amount to convert")
    from_currency: str = Field(min_length=3, max_length=3, description="ISO code, e.g. USD")
    to_currency: str = Field(min_length=3, max_length=3, description="ISO code, e.g. JPY")


@tool("convert_currency", args_schema=CurrencyInput)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict[str, Any]:
    """Convert an amount between currencies at the latest ECB reference rate. Use it to check budgets."""
    src, dst = from_currency.upper(), to_currency.upper()
    if src == dst:
        return {"amount": amount, "currency": dst, "rate": 1.0}
    try:
        data = get_json(FRANKFURTER_URL, params={"base": src, "symbols": dst, "amount": amount})
    except Exception as exc:  # noqa: BLE001
        return tool_error(exc)
    converted = (data.get("rates") or {}).get(dst)
    if converted is None:
        return {"error": f"no rate for {src}->{dst}"}
    return {"amount": converted, "currency": dst, "rate": round(converted / amount, 6), "as_of": data.get("date")}
