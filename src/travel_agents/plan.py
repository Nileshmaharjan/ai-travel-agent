"""The planner's structured output. A typed plan is what makes the eval scorers deterministic."""

from datetime import date

from pydantic import BaseModel, Field


class Money(BaseModel):
    amount: float = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3, description="ISO code, e.g. USD")


class DayPlan(BaseModel):
    date: date
    title: str
    activities: list[str]


class TripPlan(BaseModel):
    destination: str
    start_date: date
    end_date: date
    days: list[DayPlan] = Field(description="One entry per calendar day from start_date to end_date")
    estimated_total: Money | None = Field(
        description="Total estimated cost in the traveller's budget currency; null if prices were unavailable"
    )
    budget: Money | None = Field(description="The traveller's stated budget, if any")
    over_budget: bool = Field(description="True if estimated_total exceeds budget")
    sources: list[str] = Field(description="Source URLs taken from the specialists' reports")
    gaps: list[str] = Field(default_factory=list, description="Information that was unavailable")

    def to_markdown(self) -> str:
        lines = [f"# {self.destination}: {self.start_date} to {self.end_date}", ""]
        for day in self.days:
            lines.append(f"**{day.date}: {day.title}**")
            lines.extend(f"- {a}" for a in day.activities)
            lines.append("")
        if self.estimated_total:
            total = f"{self.estimated_total.amount:,.0f} {self.estimated_total.currency}"
            budget = f" (budget {self.budget.amount:,.0f} {self.budget.currency})" if self.budget else ""
            lines.append(f"**Estimated total:** {total}{budget}" + (" ⚠️ over budget" if self.over_budget else ""))
        if self.gaps:
            lines.append("**Unavailable:** " + "; ".join(self.gaps))
        if self.sources:
            lines.append("**Sources:**")
            lines.extend(f"- {s}" for s in self.sources)
        return "\n".join(lines).strip()
