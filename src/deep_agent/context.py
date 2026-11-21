"""Per-run identity passed to the agent as LangGraph runtime context."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseContext:
    """Who the run acts for. Set by the service from the authenticated session.

    Tools read the customer from here instead of taking it as an argument, so the model
    has no way to name a different account.
    """

    customer_id: str
    case_id: str
    reviewer_id: str | None = None
    """Set only on runs resumed by a human reviewer's decision."""
