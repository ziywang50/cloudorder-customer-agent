"""Actions for the coordinator: propose a refund, execute it with human approval, escalate.

propose_refund re-checks the backend rules engine and records an immutable proposal in
private graph state. It never moves money. execute_refund is the only tool that does:
every call pauses for a human reviewer (HumanInTheLoopMiddleware) who sees a review card
built from the proposal and may approve, lower the amount, or reject. After approval the
tool re-validates the call against the proposal and calls the backend with an idempotency
key derived from the case and proposal, so a retry or a resumed run cannot refund twice.
escalate_case hands work to a human team and needs no approval because it moves no money.
"""

from __future__ import annotations

import hashlib
import operator
from typing import Annotated, Any, Literal, NotRequired

from langchain.agents.middleware import AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from src.deep_agent.backend import BackendError, CloudOrderClient
from src.deep_agent.tools import (
    LINE_ID,
    RefundReason,
    Runtime,
    backend_error,
    invalid_order_id,
    to_json,
    tool_error,
)

EscalationQueue = Literal["disputes", "tier2_support", "trust_safety"]
EscalationPriority = Literal["low", "normal", "high", "urgent"]
EXECUTE_REFUND = "execute_refund"


def _merge(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    return {**(left or {}), **(right or {})}


class CaseState(AgentState):
    """Coordinator state for proposals and executed actions.

    Both channels are private: they are not part of the graph's input or output schema, so
    a caller cannot inject a proposal, and deepagents does not copy them to sub-agents.
    Only propose_refund, execute_refund, and escalate_case write to them.

    guardrails.CaseGuardrailMiddleware registers this schema (via GuardrailState) on the
    coordinator graph.
    """

    refund_proposals: NotRequired[Annotated[dict[str, dict[str, Any]], PrivateStateAttr, _merge]]
    action_receipts: NotRequired[Annotated[list[dict[str, Any]], PrivateStateAttr, operator.add]]


def usd(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def proposal_id_for(
    case_id: str, order_id: str, reason_code: str, amount_cents: int, line_ids: list[str]
) -> str:
    """Deterministic: proposing the same refund twice in a case yields the same proposal."""
    raw = "|".join([case_id, order_id, reason_code, str(amount_cents), ",".join(sorted(line_ids))])
    return "RP-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]


def refund_idempotency_key(case_id: str, proposal_id: str) -> str:
    return f"{case_id}:{proposal_id}"


def escalation_idempotency_key(case_id: str, queue: str, order_id: str | None) -> str:
    return f"{case_id}:escalation:{queue}:{order_id or 'account'}"


def refund_review_card(tool_call: dict[str, Any], state: dict[str, Any], runtime: Any) -> str:
    """What the human reviewer sees for one execute_refund call (HITL description)."""
    args = tool_call.get("args") or {}
    proposal_id = args.get("proposal_id")
    context = runtime.context
    proposal = (state.get("refund_proposals") or {}).get(proposal_id)
    header = f"Refund approval for case {context.case_id}, customer {context.customer_id}"
    if proposal is None:
        return (f"{header}\nProposal {proposal_id!r} does not exist in this case, so this call "
                "cannot be checked against the rules engine. Reject it.")
    evidence = sorted(path for path in (state.get("files") or {}) if path.startswith("/case/evidence/"))
    requested = args.get("amount_cents")
    amount = usd(requested) if isinstance(requested, int) else f"invalid ({requested!r})"
    lines = [
        header,
        f"Order {proposal['orderId']}: {proposal['reasonCode']}",
        f"Amount: {amount} (proposed {usd(proposal['amountCents'])}; rules engine allows up to "
        f"{usd(proposal['maxForReasonCents'])})",
        f"Basis: {proposal['basis']}",
        f"Agent's rationale: {proposal['rationale']}",
        f"Evidence: {', '.join(evidence) if evidence else 'none filed'}",
        "Decide: approve, edit to a lower amount, or reject with a reason.",
    ]
    return "\n".join(lines)


def build_action_tools(client: CloudOrderClient) -> dict[str, BaseTool]:
    """Build the coordinator's action tools, keyed by tool name."""

    @tool
    async def propose_refund(
        order_id: str,
        reason_code: RefundReason,
        amount_cents: int,
        rationale: str,
        runtime: Runtime,
        line_ids: list[str] | None = None,
    ) -> str | Command:
        """Propose a refund for human review. Re-checks CloudOrder's refund rules and, if the
        refund is allowed, records a proposal and returns its proposalId. This does not move
        money; call execute_refund with the proposalId next. amount_cents is an integer number
        of cents. For damaged_item_keep, pass the damaged item's line_ids. rationale is one
        sentence a reviewer can check against the evidence files."""
        if error := invalid_order_id(order_id):
            return error
        line_ids = line_ids or []
        if bad := [line for line in line_ids if not LINE_ID.fullmatch(line)]:
            return tool_error("INVALID_LINE_ID", f"Line IDs look like ORD-123456-L1; got {bad}.")
        rationale = (rationale or "").strip()
        if not 10 <= len(rationale) <= 500:
            return tool_error("INVALID_RATIONALE", "Give a one-sentence rationale (10-500 characters).")
        context = runtime.context
        try:
            preview = await client.preview_refund(
                context.customer_id, order_id=order_id, amount_cents=amount_cents,
                reason_code=reason_code, line_ids=line_ids,
            )
        except BackendError as exc:
            return backend_error(exc, order_id)
        if not preview["eligible"]:
            return to_json({
                "proposed": False,
                "blockingReasons": preview["blockingReasons"],
                "maxForReasonCents": preview["maxForReasonCents"],
                "note": "Nothing was proposed. Do not offer this refund.",
            })
        proposal_id = proposal_id_for(context.case_id, order_id, reason_code, amount_cents, line_ids)
        proposal = {
            "proposalId": proposal_id,
            "orderId": order_id,
            "reasonCode": reason_code,
            "amountCents": amount_cents,
            "lineIds": line_ids,
            "maxForReasonCents": preview["maxForReasonCents"],
            "refundableCents": preview["refundableCents"],
            "basis": preview["basis"],
            "rationale": rationale,
            "proposedAt": preview["asOf"],
        }
        content = to_json({
            "proposed": True,
            **proposal,
            "next": "Call execute_refund with this proposalId and amount. A human reviewer will "
                    "approve it, lower the amount, or reject it.",
        })
        return Command(update={
            "refund_proposals": {proposal_id: proposal},
            "messages": [ToolMessage(content, tool_call_id=runtime.tool_call_id)],
        })

    @tool
    async def execute_refund(proposal_id: str, amount_cents: int, runtime: Runtime) -> str | Command:
        """Issue a proposed refund. Every call waits for a human reviewer, who may approve it,
        lower the amount, or reject it. Pass the proposal's amountCents. The result is the only
        record of what was refunded: tell the customer its refundId and amountCents, which may
        be lower than proposed. Put all pending execute_refund calls in one message so the
        reviewer sees them together."""
        proposal = (runtime.state.get("refund_proposals") or {}).get(proposal_id)
        if proposal is None:
            return tool_error("UNKNOWN_PROPOSAL", "No proposal with that ID exists in this case. "
                                                  "Nothing was refunded. Call propose_refund first.")
        receipts = runtime.state.get("action_receipts") or []
        done = next((r for r in receipts if r.get("proposalId") == proposal_id), None)
        if done is not None:
            return to_json({"alreadyExecuted": True, "refundId": done["refundId"],
                            "amountCents": done["amountCents"],
                            "note": "This proposal was already executed; nothing new was refunded."})
        context = runtime.context
        if not context.reviewer_id:
            return tool_error("APPROVAL_MISSING", "No human approval is recorded for this run. "
                                                  "Nothing was refunded.")
        if not 0 < amount_cents <= proposal["amountCents"]:
            return tool_error("AMOUNT_ABOVE_PROPOSAL",
                              f"An approved refund can keep or lower the proposed "
                              f"{proposal['amountCents']} cents, not raise it. Nothing was refunded.")
        try:
            refund = await client.create_refund(
                context.customer_id,
                idempotency_key=refund_idempotency_key(context.case_id, proposal_id),
                order_id=proposal["orderId"],
                amount_cents=amount_cents,
                reason_code=proposal["reasonCode"],
                line_ids=proposal["lineIds"],
                case_id=context.case_id,
                approved_by=context.reviewer_id,
                note=proposal["rationale"],
            )
        except BackendError as exc:
            if exc.code == "REFUND_BLOCKED":
                reasons = "; ".join(b["message"] for b in exc.details.get("blockingReasons", []))
                return tool_error("REFUND_BLOCKED", f"The rules engine rejected this refund at "
                                                    f"execution time. Nothing was refunded. {reasons}")
            return backend_error(exc, proposal["orderId"])
        receipt = {
            "kind": "refund",
            "proposalId": proposal_id,
            "refundId": refund["refundId"],
            "orderId": refund["orderId"],
            "reasonCode": refund["reasonCode"],
            "amountCents": refund["amountCents"],
            "proposedAmountCents": proposal["amountCents"],
            "approvedBy": refund["approvedBy"],
            "status": refund["status"],
            "expectedSettlementBy": refund["expectedSettlementBy"],
        }
        visible = {key: value for key, value in receipt.items() if key != "approvedBy"}
        return Command(update={
            "action_receipts": [receipt],
            "messages": [ToolMessage(to_json({"executed": True, **visible}),
                                     tool_call_id=runtime.tool_call_id)],
        })

    @tool
    async def escalate_case(
        queue: EscalationQueue,
        priority: EscalationPriority,
        summary: str,
        runtime: Runtime,
        order_id: str | None = None,
    ) -> str | Command:
        """Hand the case to a human team. Use disputes when a chargeback is open, tier2_support
        for defects or anything policy says needs a specialist, and trust_safety for suspected
        fraud or account abuse. summary: 1-3 sentences with the facts the team needs (IDs,
        amounts, dates). No approval is needed. Returns the escalation ID and the team's
        response deadline."""
        if order_id is not None and (error := invalid_order_id(order_id)):
            return error
        summary = (summary or "").strip()
        if not 10 <= len(summary) <= 2000:
            return tool_error("INVALID_SUMMARY", "Give a 1-3 sentence summary (10-2000 characters).")
        context = runtime.context
        try:
            escalation = await client.create_escalation(
                context.customer_id,
                idempotency_key=escalation_idempotency_key(context.case_id, queue, order_id),
                queue=queue, priority=priority, summary=summary, case_id=context.case_id,
                order_id=order_id,
            )
        except BackendError as exc:
            return backend_error(exc, order_id or "")
        receipt = {
            "kind": "escalation",
            "escalationId": escalation["escalationId"],
            "queue": escalation["queue"],
            "priority": escalation["priority"],
            "orderId": escalation["orderId"],
            "firstResponseDueBy": escalation["firstResponseDueBy"],
        }
        return Command(update={
            "action_receipts": [receipt],
            "messages": [ToolMessage(to_json({"escalated": True, **receipt}),
                                     tool_call_id=runtime.tool_call_id)],
        })

    tools = [propose_refund, execute_refund, escalate_case]
    return {t.name: t for t in tools}
