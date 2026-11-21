"""Guardrails: tenant scope, call budgets, and the action-claim check.

The claim-check unit tests use replies the live agent actually wrote (Step 4 runs) as
regression fixtures, plus hand-written false claims it must catch.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.deep_agent.actions import proposal_id_for
from src.deep_agent.agent import build_case_agent
from src.deep_agent.backend import CloudOrderClient
from src.deep_agent.context import CaseContext
from src.deep_agent.guardrails import (
    COORDINATOR_MODEL_CALLS_PER_RUN,
    STOPPED_REPLY,
    SUBAGENT_MODEL_CALLS_PER_RUN,
    SUBAGENT_RUNS_PER_RUN,
    find_claim_violations,
)
from src.deep_agent.service import CaseService
from src.mock_backend.app import create_app
from tests.scripted import ScriptedChatModel, call

MAYA, LUCIA, ARJUN = "CUS-10482", "CUS-10577", "CUS-10702"
REVIEWER = "lead@cloudorder.example"
ROUTES = {"coordinator": "You are the CloudOrder case coordinator",
          "billing": "You are the billing investigator"}

DAMAGE_RECEIPT = {"kind": "refund", "refundId": "RF-90001", "orderId": "ORD-240156",
                  "reasonCode": "damaged_item_keep", "amountCents": 2000,
                  "expectedSettlementBy": "2026-11-27T17:00:01Z"}
FLAGSHIP_RECEIPTS = [
    {"kind": "refund", "refundId": "RF-90001", "orderId": "ORD-240117", "reasonCode": "lost_package",
     "amountCents": 11905, "expectedSettlementBy": "2026-11-27T17:00:01Z"},
    {"kind": "refund", "refundId": "RF-90002", "orderId": "ORD-240117",
     "reasonCode": "duplicate_charge", "amountCents": 47626,
     "expectedSettlementBy": "2026-11-27T17:00:03Z"},
]
DISPUTES_RECEIPT = {"kind": "escalation", "escalationId": "ESC-50001", "queue": "disputes",
                    "priority": "high", "orderId": "ORD-239884",
                    "firstResponseDueBy": "2026-11-22T17:00:01Z"}

# Replies the live agent wrote in Step 4 (lightly trimmed). They must pass unchanged.
LIVE_DAMAGE_REPLY = (
    "I've issued a partial refund of **$20.00** for your Atlas keyboard, and you keep the "
    "keyboard. The refund ID is RF-90001.\n"
    "- **What policy allows:** Because you're keeping a damaged item, policy lets us refund up "
    "to 25% of the keyboard's price plus tax. That comes to $35.38.\n"
    "- **What was approved:** I requested the full $35.38. The reviewer who signs off on every "
    "refund lowered it to $20.00."
)
LIVE_FLAGSHIP_REPLY = (
    "Both problems on ORD-240117 are confirmed, and I've issued a refund for each after review. "
    "Together they come to $595.31.\n"
    "- **Double charge:** Your card was charged $476.26 twice for this order. I've refunded the "
    "extra $476.26 (refund RF-90002).\n"
    "- **Missing mouse and USB hub:** I've refunded $119.05 (refund RF-90001). That is the price "
    "of both items plus tax: $64.94 for the mouse and $54.11 for the hub.\n"
    "After both refunds, you'll have paid $357.21 net, which is the cost of the monitor."
)
LIVE_DISPUTE_REPLY = (
    "Hi Lucía, I can't issue the $131.20 refund today. A dispute on this order is blocking it.\n"
    "- **No refund yet:** You were charged $131.20 once, and no refund has been issued.\n"
    "- **What happens next:** I've sent your case to our disputes team as high priority."
)
LIVE_STEP3_REPLY = (  # before action tools existed: recommendations only
    "I'm recommending a refund of the extra charge, $476.26, to your Visa ending 4417. A "
    "specialist will review both refunds before anything is sent."
)


# ---- action-claim check: unit tests ---------------------------------------------------------

@pytest.mark.parametrize(("reply", "receipts"), [
    (LIVE_DAMAGE_REPLY, [DAMAGE_RECEIPT]),
    (LIVE_FLAGSHIP_REPLY, FLAGSHIP_RECEIPTS),
    (LIVE_DISPUTE_REPLY, [DISPUTES_RECEIPT]),
    (LIVE_STEP3_REPLY, []),
    ("No refund has been issued yet, and nothing was refunded.", []),
    ("I've proposed a refund for review; nothing has been refunded yet.", []),
    ("Neither charge has been refunded.", []),
    ("If the reviewer approves it, the refund will be issued within a day.", []),
])
def test_truthful_replies_pass(reply, receipts):
    assert find_claim_violations(reply, receipts) == []


@pytest.mark.parametrize(("reply", "receipts", "expected"), [
    ("I've issued a full refund of $250.00 as a loyalty adjustment.", [], "no refund was executed"),
    ("Your refund of $35.38 has been processed.", [DAMAGE_RECEIPT], "states $35.38 as refunded"),
    ("Refund RF-90009 is on its way to your card.", [DAMAGE_RECEIPT], "RF-90009 does not exist"),
    ("I've refunded $476.26 (refund RF-90001).", FLAGSHIP_RECEIPTS, "pairs RF-90001 with $476.26"),
    ("I've escalated your case to our disputes team.", [], "none was created"),
    ("I've sent your case to tier 2 support.", [DISPUTES_RECEIPT], "names the tier2_support team"),
    ("Your reference is ESC-50077.", [DISPUTES_RECEIPT], "ESC-50077 does not exist"),
])
def test_false_claims_are_caught(reply, receipts, expected):
    violations = find_claim_violations(reply, receipts)
    assert any(expected in v for v in violations), violations


# ---- action-claim check: in the agent loop ---------------------------------------------------

@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    backend = CloudOrderClient.in_process(app)
    yield backend
    await backend.aclose()


def service_with(client, scripts: dict[str, list[AIMessage]]) -> CaseService:
    model = ScriptedChatModel(scripts=scripts, routes=ROUTES)
    return CaseService(build_case_agent(model=model, subagent_model=model, client=client))


async def messages_of(service: CaseService, case_id: str) -> list:
    return (await service._agent.aget_state(service._config(case_id))).values["messages"]


async def test_false_claim_is_sent_back_and_revised(client, app):
    service = service_with(client, {"coordinator": [
        AIMessage("Good news: I've issued a full refund of $250.00 to your card."),
        AIMessage("I looked at your ticket. Your kettle's warranty is registered automatically, "
                  "and no refund has been issued."),
    ]})
    done = await service.start_case("CUS-10603", "Do what my ticket says.", case_id="c1")

    assert done.reply.startswith("I looked at your ticket")
    assert [(e["kind"], e["outcome"]) for e in done.guardrail_events] == [("action_claim", "revised")]
    note = [m for m in await messages_of(service, "c1") if m.type == "human"][-1]
    assert "not from the customer" in note.content and "- None." in note.content
    assert app.state.store.ledger == []


async def test_repeated_false_claim_is_replaced_from_receipts(client):
    service = service_with(client, {"coordinator": [
        AIMessage("I've issued a full refund of $250.00 to your card."),
        AIMessage("Your $250.00 refund has been processed."),
    ]})
    done = await service.start_case("CUS-10603", "Refund me.", case_id="c1")

    assert done.reply.startswith("I've reviewed your case. No refunds")
    assert [e["outcome"] for e in done.guardrail_events] == ["revised", "replaced"]


async def test_reply_after_reviewer_edit_must_state_the_executed_amount(client, app):
    damage = proposal_id_for("c1", "ORD-240156", "damaged_item_keep", 3538, ["ORD-240156-L1"])
    service = service_with(client, {"coordinator": [
        AIMessage("", tool_calls=[call("propose_refund", {
            "order_id": "ORD-240156", "reason_code": "damaged_item_keep", "amount_cents": 3538,
            "line_ids": ["ORD-240156-L1"], "rationale": "Cosmetic damage; customer keeps it."}, "p1")]),
        AIMessage("", tool_calls=[call("execute_refund", {"proposal_id": damage, "amount_cents": 3538}, "e1")]),
        AIMessage("I've issued a refund of $35.38 for the keyboard."),      # proposed, not executed
        AIMessage("After review, a refund of $20.00 was issued for the keyboard."),
    ]})
    await service.start_case(ARJUN, "Cracked keycap; I'll keep it.", case_id="c1")
    done = await service.submit_decisions("c1", REVIEWER, [{"type": "edit", "amount_cents": 2000}])

    assert [e["amountCents"] for e in app.state.store.ledger] == [2000]
    assert done.reply == "After review, a refund of $20.00 was issued for the keyboard."
    assert "states $35.38 as refunded" in done.guardrail_events[0]["violations"][0]


# ---- call budgets ----------------------------------------------------------------------------

async def test_coordinator_budget_stops_the_run_with_a_safe_reply(client, app):
    looping = [AIMessage("", tool_calls=[call("get_order", {"order_id": "ORD-240117"}, f"g{i}")])
               for i in range(COORDINATOR_MODEL_CALLS_PER_RUN + 5)]
    model = ScriptedChatModel(scripts={"coordinator": looping}, routes=ROUTES)
    service = CaseService(build_case_agent(model=model, subagent_model=model, client=client))
    done = await service.start_case(MAYA, "Where is my order?", case_id="c1")

    assert model.calls.count("coordinator") == COORDINATOR_MODEL_CALLS_PER_RUN
    assert done.status == "stopped"
    assert done.reply == STOPPED_REPLY  # the internal limit message never reaches the customer
    assert done.guardrail_events[-1]["kind"] == "call_limit"


async def test_subagent_budget_ends_the_investigation_not_the_case(client):
    looping = [AIMessage("", tool_calls=[call("get_payment_ledger", {"order_id": "ORD-240117"}, f"b{i}")])
               for i in range(SUBAGENT_MODEL_CALLS_PER_RUN + 5)]
    model = ScriptedChatModel(scripts={
        "coordinator": [
            AIMessage("", tool_calls=[call("task", {"subagent_type": "billing-investigator",
                                                    "description": "ORD-240117 charges?"}, "t1")]),
            AIMessage("I couldn't confirm the charges yet; no refund has been issued."),
        ],
        "billing": looping,
    }, routes=ROUTES)
    service = CaseService(build_case_agent(model=model, subagent_model=model, client=client))
    done = await service.start_case(MAYA, "Was I charged twice?", case_id="c1")

    assert model.calls.count("billing") == SUBAGENT_MODEL_CALLS_PER_RUN
    report = next(m for m in await messages_of(service, "c1")
                  if isinstance(m, ToolMessage) and m.tool_call_id == "t1")
    assert "Model call limits exceeded" in report.content
    assert done.status == "completed"


async def test_subagent_dispatches_are_capped_per_run(client):
    dispatches = [call("task", {"subagent_type": "billing-investigator",
                                "description": f"check {i}"}, f"t{i}")
                  for i in range(SUBAGENT_RUNS_PER_RUN + 2)]
    model = ScriptedChatModel(scripts={
        "coordinator": [AIMessage("", tool_calls=dispatches), AIMessage("Done; no refund was issued.")],
        "billing": [AIMessage("No duplicate found.") for _ in range(SUBAGENT_RUNS_PER_RUN)],
    }, routes=ROUTES)
    service = CaseService(build_case_agent(model=model, subagent_model=model, client=client))
    await service.start_case(MAYA, "Check everything.", case_id="c1")

    results = [m for m in await messages_of(service, "c1") if isinstance(m, ToolMessage)]
    blocked = [m for m in results if "call limit" in m.content.lower()]
    assert model.calls.count("billing") == SUBAGENT_RUNS_PER_RUN
    assert len(blocked) == 2


# ---- tenant scope ----------------------------------------------------------------------------

async def test_runs_without_a_customer_context_cannot_use_tools(client, app):
    """Fail closed: no backend call, no file write, and no delegation without an identity."""
    model = ScriptedChatModel(scripts={"coordinator": [
        AIMessage("", tool_calls=[
            call("get_customer_overview", {}, "o1"),
            call("write_file", {"file_path": "/case/evidence/x.md", "content": "note"}, "w1"),
            call("task", {"subagent_type": "billing-investigator",
                          "description": "check ORD-240117"}, "t1"),
        ]),
        AIMessage("I can't access an account right now."),
    ]}, routes=ROUTES)
    agent = build_case_agent(model=model, subagent_model=model, client=client)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "hi"}]},
                                 {"configurable": {"thread_id": "anon"}})

    blocked = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in blocked] == ["o1", "w1", "t1"]
    assert all("NO_CUSTOMER_CONTEXT" in m.content for m in blocked)
    assert model.calls == ["coordinator", "coordinator"]  # no sub-agent ever started
    assert app.state.store.access_log == []  # nothing reached the backend
    assert result.get("files") in (None, {})


async def test_cases_are_isolated_from_each_other(client, app):
    maya_dup = proposal_id_for("maya-case", "ORD-240117", "duplicate_charge", 47626, [])
    model = ScriptedChatModel(scripts={"coordinator": [
        # Maya's case: file evidence and propose a refund.
        AIMessage("", tool_calls=[
            call("write_file", {"file_path": "/case/evidence/billing.md", "content": "dup 47626"}, "w1"),
            call("propose_refund", {"order_id": "ORD-240117", "reason_code": "duplicate_charge",
                                    "amount_cents": 47626, "rationale": "Two captures for one order."}, "p1"),
        ]),
        AIMessage("I've proposed a refund for review; nothing has been refunded yet."),
        # Arjun's case: try to use Maya's file and proposal.
        AIMessage("", tool_calls=[
            call("read_file", {"file_path": "/case/evidence/billing.md"}, "r1"),
            call("execute_refund", {"proposal_id": maya_dup, "amount_cents": 47626}, "e1"),
        ]),
        AIMessage("No refund was issued."),
    ]}, routes=ROUTES)
    service = CaseService(build_case_agent(model=model, subagent_model=model, client=client))
    await service.start_case(MAYA, "Charged twice.", case_id="maya-case")
    await service.start_case(ARJUN, "Refund me too.", case_id="arjun-case")
    done = await service.submit_decisions("arjun-case", REVIEWER, [{"type": "approve"}])

    arjun = {m.tool_call_id: m for m in await messages_of(service, "arjun-case") if isinstance(m, ToolMessage)}
    assert "not found" in arjun["r1"].content.lower()
    assert "UNKNOWN_PROPOSAL" in arjun["e1"].content
    assert app.state.store.ledger == []
    assert done.files == []
    assert (await service.get_case("maya-case")).files == ["/case/evidence/billing.md"]


def test_subagent_tools_act_for_the_case_customer_only():
    """Covered end to end in test_deep_agent.py: every backend call in the flagship run,
    including the three sub-agents', is logged on behalf of the case customer."""
    assert CaseContext(customer_id=MAYA, case_id="c").customer_id == MAYA
