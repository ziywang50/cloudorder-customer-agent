"""Runtime guardrails for the case agent.

1. Tenant scope: every tool call fails closed unless the run carries an authenticated
   CaseContext. Tools already take the customer from that context (never from arguments)
   and the backend enforces it again with X-On-Behalf-Of; this closes the remaining gap of
   a run started without an identity.
2. Call budgets: built-in model- and tool-call limits bound every run. When the
   coordinator runs out, the internal "limits exceeded" text never reaches the customer;
   it is replaced with a safe reply and the case is marked stopped.
3. Action-claim check: before a final reply is accepted, every refund or escalation it
   claims is compared with the receipts of tools that actually ran. A reply that claims
   an action that did not happen, quotes an unexecuted amount, or cites an unknown refund
   or escalation ID is sent back once for revision, then replaced with a reply generated
   from the receipts.
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.middleware.types import PrivateStateAttr, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.deep_agent.actions import EXECUTE_REFUND, CaseState, usd
from src.deep_agent.context import CaseContext
from src.deep_agent.tools import tool_error

# ---- call budgets ------------------------------------------------------------------------
# Worst case per run: 20 coordinator model calls + 6 sub-agent runs x 10 model calls = 80.

COORDINATOR_MODEL_CALLS_PER_RUN = 20
COORDINATOR_MODEL_CALLS_PER_CASE = 40
COORDINATOR_TOOL_CALLS_PER_RUN = 40
SUBAGENT_RUNS_PER_RUN = 6
REFUND_EXECUTIONS_PER_CASE = 4
SUBAGENT_MODEL_CALLS_PER_RUN = 10
SUBAGENT_TOOL_CALLS_PER_RUN = 20
MAX_CLAIM_REVISIONS = 1

_LIMIT_MESSAGE_PREFIX = "Model call limits exceeded"
STOPPED_REPLY = (
    "I wasn't able to finish reviewing your case automatically, so it needs a support "
    "specialist to take it from here."
)


def coordinator_limits() -> list[AgentMiddleware]:
    return [
        ModelCallLimitMiddleware(thread_limit=COORDINATOR_MODEL_CALLS_PER_CASE,
                                 run_limit=COORDINATOR_MODEL_CALLS_PER_RUN, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=COORDINATOR_TOOL_CALLS_PER_RUN),
        ToolCallLimitMiddleware(tool_name="task", run_limit=SUBAGENT_RUNS_PER_RUN),
        ToolCallLimitMiddleware(tool_name=EXECUTE_REFUND, thread_limit=REFUND_EXECUTIONS_PER_CASE),
    ]


def subagent_limits() -> list[AgentMiddleware]:
    # A sub-agent that runs out ends with the limit message as its report; the coordinator
    # sees that the investigation stopped early instead of the whole case failing.
    return [
        TenantScopeMiddleware(),
        ModelCallLimitMiddleware(run_limit=SUBAGENT_MODEL_CALLS_PER_RUN, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=SUBAGENT_TOOL_CALLS_PER_RUN),
    ]


# ---- tenant scope --------------------------------------------------------------------------

class TenantScopeMiddleware(AgentMiddleware):
    """Refuse every tool call in a run that has no authenticated customer."""

    async def awrap_tool_call(self, request, handler):
        context = request.runtime.context
        if not isinstance(context, CaseContext) or not context.customer_id or not context.case_id:
            return ToolMessage(
                tool_error("NO_CUSTOMER_CONTEXT",
                           "This run has no authenticated customer, so no tool can run."),
                tool_call_id=request.tool_call["id"], name=request.tool_call["name"],
                status="error",
            )
        return await handler(request)


# ---- action-claim check --------------------------------------------------------------------

_MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?")
_REFUND_ID = re.compile(r"\bRF-\d{5}\b")
_ESCALATION_ID = re.compile(r"\bESC-\d{5}\b")
_CLAUSE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+|\s+-\s+|\s+—\s+")
_REFUND_WORD = re.compile(r"\brefund", re.IGNORECASE)
_REFUND_DONE = re.compile(
    r"\b(refunded|issued|processed|credited|approved|sent back|returned to your (?:card|account))\b",
    re.IGNORECASE,
)
_ESCALATION_WORD = re.compile(
    r"\b(escalat\w*|disputes team|specialist team|tier ?2|trust (?:&|and) safety|"
    r"support team|specialist)\b",
    re.IGNORECASE,
)
_ESCALATION_DONE = re.compile(
    r"\b(escalated|sent|passed|handed|forwarded|referred|routed|notified|"
    r"opened an? (?:case|ticket))\b",
    re.IGNORECASE,
)
# Words before the action verb that make a clause conditional or negative, not a claim.
_NOT_A_CLAIM = re.compile(
    r"\b(not|no|nothing|none|neither|nor|without|never|cannot|unable|yet|if|once|unless|until|"
    r"before|after|would|could|should|might|may|will|can|recommend\w*|propos\w*|request\w*|"
    r"ask\w*|eligible)\b|n't\b",
    re.IGNORECASE,
)
_QUEUE_WORDS = {"disputes": re.compile(r"\bdisputes?\b", re.IGNORECASE),
                "tier2_support": re.compile(r"\btier ?2\b", re.IGNORECASE),
                "trust_safety": re.compile(r"\btrust (?:&|and) safety\b", re.IGNORECASE)}


def _cents(match: re.Match[str]) -> int:
    return int(match.group(1).replace(",", "")) * 100 + int(match.group(2) or 0)


def _is_claim(clause: str, verb: re.Pattern[str]) -> bool:
    """True when the clause states the action as done, not negated or conditional."""
    match = verb.search(clause)
    return bool(match) and not _NOT_A_CLAIM.search(clause[: match.start()])


def find_claim_violations(reply: str, receipts: list[dict[str, Any]]) -> list[str]:
    """Problems with what `reply` says was done, judged against executed-action receipts."""
    text = reply.replace("**", "").replace("__", "")
    refunds = [r for r in receipts if r.get("kind") == "refund"]
    escalations = [r for r in receipts if r.get("kind") == "escalation"]
    refund_ids = {r["refundId"] for r in refunds}
    escalation_ids = {r["escalationId"] for r in escalations}
    amounts = {r["amountCents"] for r in refunds}
    if refunds:
        amounts.add(sum(r["amountCents"] for r in refunds))

    violations: list[str] = []
    for refund_id in sorted(set(_REFUND_ID.findall(text)) - refund_ids):
        violations.append(f"Refund ID {refund_id} does not exist in this case.")
    for escalation_id in sorted(set(_ESCALATION_ID.findall(text)) - escalation_ids):
        violations.append(f"Escalation ID {escalation_id} does not exist in this case.")

    for clause in filter(None, (c.strip() for c in _CLAUSE_SPLIT.split(text))):
        if _REFUND_WORD.search(clause) and _is_claim(clause, _REFUND_DONE):
            if not refunds:
                violations.append(f'"{clause}" claims a refund, but no refund was executed.')
                continue
            stated = [_cents(match) for match in _MONEY.finditer(clause)]
            for cents in stated:
                if cents not in amounts:
                    violations.append(
                        f'"{clause}" states {usd(cents)} as refunded, but the executed refunds '
                        f"are {', '.join(usd(a) for a in sorted(amounts))}."
                    )
            named = [r for r in refunds if r["refundId"] in clause]
            if len(named) == 1 and len(stated) == 1 and stated[0] in amounts \
                    and stated[0] != named[0]["amountCents"]:
                violations.append(f'"{clause}" pairs {named[0]["refundId"]} with {usd(stated[0])}, '
                                  f"but that refund is {usd(named[0]['amountCents'])}.")
        if _ESCALATION_WORD.search(clause) and _is_claim(clause, _ESCALATION_DONE):
            if not escalations:
                violations.append(f'"{clause}" claims an escalation, but none was created.')
                continue
            queues = {e["queue"] for e in escalations}
            for queue, pattern in _QUEUE_WORDS.items():
                if pattern.search(clause) and queue not in queues:
                    violations.append(f'"{clause}" names the {queue} team, but the case was '
                                      f"escalated to {', '.join(sorted(queues))}.")
    return violations


def describe_receipts(receipts: list[dict[str, Any]]) -> str:
    lines = []
    for receipt in receipts:
        if receipt.get("kind") == "refund":
            lines.append(f"- Refund {receipt['refundId']}: {usd(receipt['amountCents'])} on order "
                         f"{receipt['orderId']} ({receipt['reasonCode']}), expected to settle by "
                         f"{receipt['expectedSettlementBy'][:10]}.")
        elif receipt.get("kind") == "escalation":
            lines.append(f"- Escalation {receipt['escalationId']}: sent to the {receipt['queue']} "
                         f"team, first response due by {receipt['firstResponseDueBy'][:10]}.")
    return "\n".join(lines) if lines else "- None."


def receipts_reply(receipts: list[dict[str, Any]]) -> str:
    """A reply built only from receipts, used when a draft fails the check twice."""
    refunds = [r for r in receipts if r.get("kind") == "refund"]
    escalations = [r for r in receipts if r.get("kind") == "escalation"]
    if not refunds and not escalations:
        return ("I've reviewed your case. No refunds or other changes have been made to your "
                "account in this conversation.")
    lines = ["Here is where your case stands:"]
    for refund in refunds:
        lines.append(f"- A refund of {usd(refund['amountCents'])} (ID {refund['refundId']}) was "
                     f"issued for order {refund['orderId']}. It should reach your original "
                     f"payment method by {refund['expectedSettlementBy'][:10]}.")
    for escalation in escalations:
        lines.append(f"- Your case was sent to our {escalation['queue'].replace('_', ' ')} team "
                     f"(reference {escalation['escalationId']}). They will respond by "
                     f"{escalation['firstResponseDueBy'][:10]}.")
    if not refunds:
        lines.append("No refunds have been issued.")
    return "\n".join(lines)


class GuardrailState(CaseState):
    guardrail_events: NotRequired[Annotated[list[dict[str, Any]], PrivateStateAttr, operator.add]]
    claim_revisions: NotRequired[Annotated[int, PrivateStateAttr]]
    stopped_reason: NotRequired[Annotated[str | None, PrivateStateAttr]]


class CaseGuardrailMiddleware(TenantScopeMiddleware):
    """Coordinator guardrails: tenant scope, action-claim check, and safe budget stops."""

    state_schema = GuardrailState

    async def abefore_agent(self, state, runtime) -> dict[str, Any] | None:
        return {"stopped_reason": None} if state.get("stopped_reason") else None

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime) -> dict[str, Any] | None:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        receipts = state.get("action_receipts") or []
        violations = find_claim_violations(last.text, receipts)
        revisions = state.get("claim_revisions") or 0
        if not violations:
            return {"claim_revisions": 0} if revisions else None

        event = {"kind": "action_claim", "violations": violations}
        if revisions < MAX_CLAIM_REVISIONS:
            note = (
                "[Automated action check. This message is not from the customer.] Your draft "
                "reply describes actions the case record does not support:\n"
                + "\n".join(f"- {v}" for v in violations)
                + "\nActions actually executed in this case:\n" + describe_receipts(receipts)
                + "\nRewrite the reply so every refund or escalation you mention matches these "
                "records exactly. If an action you described is still needed, take it with the "
                "tool first; otherwise remove the claim."
            )
            return {
                "jump_to": "model",
                "claim_revisions": revisions + 1,
                "guardrail_events": [{**event, "outcome": "revised"}],
                "messages": [HumanMessage(content=note)],
            }
        return {
            "claim_revisions": 0,
            "guardrail_events": [{**event, "outcome": "replaced"}],
            "messages": [AIMessage(content=receipts_reply(receipts), id=last.id)],
        }

    async def aafter_agent(self, state, runtime) -> dict[str, Any] | None:
        last = state["messages"][-1] if state["messages"] else None
        if isinstance(last, AIMessage) and last.text.startswith(_LIMIT_MESSAGE_PREFIX):
            return {
                "stopped_reason": "call_limit",
                "guardrail_events": [{"kind": "call_limit", "outcome": "stopped",
                                      "detail": last.text}],
                "messages": [AIMessage(content=STOPPED_REPLY, id=last.id)],
            }
        return None
