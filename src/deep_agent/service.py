"""Case lifecycle around the deep agent: start a case, pause for refund approvals, resume
with a reviewer's decisions.

The service owns case identity. The customer for a case is recorded when the case starts
and reused on every resume, so a reviewer's request can never change whose account the
agent acts on. Only `submit_decisions` resumes a paused case, and it always carries the
reviewer's ID, which the backend stores as `approvedBy` on every refund.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.deep_agent.actions import EXECUTE_REFUND
from src.deep_agent.context import CaseContext

CaseStatus = Literal["awaiting_approval", "completed", "stopped"]


@dataclass(frozen=True)
class ApprovalRequest:
    """One gated tool call waiting for a reviewer."""

    index: int
    tool: str
    args: dict[str, Any]
    proposal: dict[str, Any] | None
    description: str
    allowed_decisions: list[str]


@dataclass(frozen=True)
class CaseSnapshot:
    case_id: str
    customer_id: str
    status: CaseStatus
    reply: str | None
    approvals: list[ApprovalRequest] = field(default_factory=list)
    todos: list[dict[str, Any]] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    receipts: list[dict[str, Any]] = field(default_factory=list)
    guardrail_events: list[dict[str, Any]] = field(default_factory=list)


class CaseError(ValueError):
    """The request does not fit the case's current state."""


class CaseService:
    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self._customers: dict[str, str] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @staticmethod
    def _config(case_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": case_id}}

    def _customer_of(self, case_id: str) -> str:
        try:
            return self._customers[case_id]
        except KeyError:
            raise CaseError(f"Unknown case {case_id}.") from None

    async def start_case(self, customer_id: str, message: str, case_id: str | None = None) -> CaseSnapshot:
        """Run a new case until it finishes or pauses for approval."""
        case_id = case_id or f"case-{uuid.uuid4().hex[:12]}"
        if case_id in self._customers:
            raise CaseError(f"Case {case_id} already exists.")
        self._customers[case_id] = customer_id
        async with self._locks[case_id]:
            await self._agent.ainvoke(
                {"messages": [HumanMessage(content=message)]},
                self._config(case_id),
                context=CaseContext(customer_id=customer_id, case_id=case_id),
            )
        return await self.get_case(case_id)

    async def get_case(self, case_id: str) -> CaseSnapshot:
        customer_id = self._customer_of(case_id)
        state = await self._agent.aget_state(self._config(case_id))
        values = state.values
        proposals = values.get("refund_proposals") or {}
        approvals: list[ApprovalRequest] = []
        for pending in state.interrupts:
            value = pending.value
            for index, (request, review) in enumerate(
                zip(value["action_requests"], value["review_configs"], strict=True)
            ):
                args = dict(request["args"])
                approvals.append(ApprovalRequest(
                    index=index,
                    tool=request["name"],
                    args=args,
                    proposal=proposals.get(args.get("proposal_id")),
                    description=request.get("description", ""),
                    allowed_decisions=list(review["allowed_decisions"]),
                ))
        reply = None
        if not approvals:
            last = next((m for m in reversed(values.get("messages", [])) if isinstance(m, AIMessage)), None)
            reply = last.text if last is not None else None
        if approvals:
            status: CaseStatus = "awaiting_approval"
        elif values.get("stopped_reason"):
            status = "stopped"
        else:
            status = "completed"
        return CaseSnapshot(
            case_id=case_id,
            customer_id=customer_id,
            status=status,
            reply=reply,
            approvals=approvals,
            todos=list(values.get("todos") or []),
            files=sorted(values.get("files") or {}),
            receipts=list(values.get("action_receipts") or []),
            guardrail_events=list(values.get("guardrail_events") or []),
        )

    async def submit_decisions(
        self, case_id: str, reviewer_id: str, decisions: list[dict[str, Any]]
    ) -> CaseSnapshot:
        """Resume a paused case with one decision per pending approval, in order.

        Decisions: {"type": "approve"}, {"type": "reject", "message": "..."}, or
        {"type": "edit", "amount_cents": 2000}. An edit can only change the amount; the
        proposal it refers to stays fixed.
        """
        if not reviewer_id or not reviewer_id.strip():
            raise CaseError("A reviewer ID is required to approve or reject refunds.")
        customer_id = self._customer_of(case_id)
        async with self._locks[case_id]:
            snapshot = await self.get_case(case_id)
            if snapshot.status != "awaiting_approval":
                raise CaseError(f"Case {case_id} is not waiting for approval.")
            if len(decisions) != len(snapshot.approvals):
                raise CaseError(f"Expected {len(snapshot.approvals)} decision(s), got {len(decisions)}.")
            resume = [_to_hitl_decision(d, a) for d, a in zip(decisions, snapshot.approvals, strict=True)]
            await self._agent.ainvoke(
                Command(resume={"decisions": resume}),
                self._config(case_id),
                context=CaseContext(customer_id=customer_id, case_id=case_id, reviewer_id=reviewer_id),
            )
        return await self.get_case(case_id)


def _to_hitl_decision(decision: dict[str, Any], approval: ApprovalRequest) -> dict[str, Any]:
    kind = decision.get("type")
    if kind not in approval.allowed_decisions:
        raise CaseError(f"Decision {kind!r} is not allowed for {approval.tool}.")
    if kind == "approve":
        return {"type": "approve"}
    if kind == "reject":
        return {"type": "reject", "message": decision.get("message") or "Rejected by the reviewer."}
    if kind == "edit":
        amount = decision.get("amount_cents")
        if approval.tool != EXECUTE_REFUND or not isinstance(amount, int) or amount <= 0:
            raise CaseError("An edit must give a positive integer amount_cents for a refund.")
        return {"type": "edit",
                "edited_action": {"name": approval.tool, "args": {**approval.args, "amount_cents": amount}}}
    raise CaseError(f"Unknown decision type {kind!r}.")
