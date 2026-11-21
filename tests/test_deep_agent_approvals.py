"""Propose-then-execute with human approval, driven through CaseService.

A scripted coordinator proposes and executes refunds; the tests play the reviewer. Every
assertion about money is made against the backend ledger, not the agent's own messages.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.deep_agent.actions import proposal_id_for
from src.deep_agent.agent import build_case_agent
from src.deep_agent.backend import CloudOrderClient
from src.deep_agent.service import CaseError, CaseService
from src.mock_backend.app import create_app
from tests.scripted import ScriptedChatModel, call

MAYA, LUCIA, ARJUN = "CUS-10482", "CUS-10577", "CUS-10702"
REVIEWER = "lead@cloudorder.example"
CASE = "case-test"
DUP = proposal_id_for(CASE, "ORD-240117", "duplicate_charge", 47626, [])
LOST = proposal_id_for(CASE, "ORD-240117", "lost_package", 11905, [])
DAMAGE = proposal_id_for(CASE, "ORD-240156", "damaged_item_keep", 3538, ["ORD-240156-L1"])


def propose(call_id: str, order_id: str, reason: str, amount: int, line_ids=None) -> dict:
    args = {"order_id": order_id, "reason_code": reason, "amount_cents": amount,
            "rationale": f"Evidence supports a {reason} refund of {amount} cents."}
    if line_ids:
        args["line_ids"] = line_ids
    return call("propose_refund", args, call_id)


def execute(call_id: str, proposal_id: str, amount: int) -> dict:
    return call("execute_refund", {"proposal_id": proposal_id, "amount_cents": amount}, call_id)


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    backend = CloudOrderClient.in_process(app)
    yield backend
    await backend.aclose()


def make_service(client, coordinator_replies: list[AIMessage]) -> CaseService:
    model = ScriptedChatModel(scripts={"coordinator": coordinator_replies},
                              routes={"coordinator": "You are the CloudOrder case coordinator"})
    return CaseService(build_case_agent(model=model, subagent_model=model, client=client))


def tool_results(snapshot_values: dict) -> dict[str, ToolMessage]:
    return {m.tool_call_id: m for m in snapshot_values["messages"] if isinstance(m, ToolMessage)}


async def state_of(service: CaseService, case_id: str = CASE) -> dict:
    return (await service._agent.aget_state(service._config(case_id))).values


async def test_flagship_refunds_wait_for_approval_then_execute(client, app):
    service = make_service(client, [
        AIMessage("", tool_calls=[propose("p1", "ORD-240117", "duplicate_charge", 47626),
                                  propose("p2", "ORD-240117", "lost_package", 11905)]),
        AIMessage("", tool_calls=[execute("e1", DUP, 47626), execute("e2", LOST, 11905)]),
        AIMessage("Refunds RF-90001 ($476.26) and RF-90002 ($119.05) were issued."),
    ])
    paused = await service.start_case(MAYA, "Charged twice and a box is missing.", case_id=CASE)

    assert paused.status == "awaiting_approval"
    assert [a.args["proposal_id"] for a in paused.approvals] == [DUP, LOST]  # one batched review
    assert paused.approvals[0].proposal["maxForReasonCents"] == 47626
    assert ("Amount: $476.26 (proposed $476.26; rules engine allows up to $476.26)"
            in paused.approvals[0].description)
    assert paused.approvals[0].allowed_decisions == ["approve", "edit", "reject"]
    assert app.state.store.ledger == []  # nothing moves before a human decides

    done = await service.submit_decisions(CASE, REVIEWER, [{"type": "approve"}, {"type": "approve"}])
    assert done.status == "completed"
    ledger = app.state.store.ledger
    assert [(e["reasonCode"], e["amountCents"], e["approvedBy"], e["caseId"]) for e in ledger] == [
        ("duplicate_charge", 47626, REVIEWER, CASE),
        ("lost_package", 11905, REVIEWER, CASE),
    ]
    assert [r["refundId"] for r in done.receipts] == ["RF-90001", "RF-90002"]
    assert {e["onBehalfOf"] for e in app.state.store.access_log} == {MAYA}
    assert done.reply.startswith("Refunds RF-90001")


async def test_reviewer_can_lower_the_amount(client, app):
    service = make_service(client, [
        AIMessage("", tool_calls=[propose("p1", "ORD-240156", "damaged_item_keep", 3538, ["ORD-240156-L1"])]),
        AIMessage("", tool_calls=[execute("e1", DAMAGE, 3538)]),
        AIMessage("A $20.00 partial refund (RF-90001) was issued."),
    ])
    await service.start_case(ARJUN, "Cracked keycap; I'll keep it.", case_id=CASE)
    done = await service.submit_decisions(CASE, REVIEWER, [{"type": "edit", "amount_cents": 2000}])

    assert [e["amountCents"] for e in app.state.store.ledger] == [2000]
    result = tool_results(await state_of(service))["e1"]
    assert '"amountCents":2000' in result.content and '"proposedAmountCents":3538' in result.content
    assert "human reviewer replaced this tool call" in result.content  # edit notice for the model
    assert done.receipts[0]["amountCents"] == 2000


async def test_reviewer_edit_cannot_raise_the_amount(client, app):
    service = make_service(client, [
        AIMessage("", tool_calls=[propose("p1", "ORD-240156", "damaged_item_keep", 3538, ["ORD-240156-L1"])]),
        AIMessage("", tool_calls=[execute("e1", DAMAGE, 3538)]),
        AIMessage("No refund was issued."),
    ])
    await service.start_case(ARJUN, "Cracked keycap.", case_id=CASE)
    done = await service.submit_decisions(CASE, REVIEWER, [{"type": "edit", "amount_cents": 5000}])

    assert "AMOUNT_ABOVE_PROPOSAL" in tool_results(await state_of(service))["e1"].content
    assert app.state.store.ledger == []
    assert done.receipts == []


async def test_rejected_refund_moves_no_money(client, app):
    service = make_service(client, [
        AIMessage("", tool_calls=[propose("p1", "ORD-240117", "duplicate_charge", 47626)]),
        AIMessage("", tool_calls=[execute("e1", DUP, 47626)]),
        AIMessage("Your refund request was not approved yet; a specialist will follow up."),
    ])
    await service.start_case(MAYA, "Charged twice.", case_id=CASE)
    await service.submit_decisions(CASE, REVIEWER, [
        {"type": "reject", "message": "The processor already reversed the second capture."},
    ])

    result = tool_results(await state_of(service))["e1"]
    assert result.status == "error"
    assert "already reversed" in result.content
    assert app.state.store.ledger == []


async def test_open_chargeback_is_not_proposed_and_escalates_without_approval(client, app):
    service = make_service(client, [
        AIMessage("", tool_calls=[propose("p1", "ORD-239884", "damaged_item_keep", 3000, ["ORD-239884-L1"])]),
        AIMessage("", tool_calls=[call("escalate_case", {
            "queue": "disputes", "priority": "high", "order_id": "ORD-239884",
            "summary": "Chargeback dp_1QK2ZrT9 is open on ORD-239884; refund frozen per policy.",
        }, "x1")]),
        AIMessage("Your bank's dispute is in progress, and our disputes team has the case."),
    ])
    done = await service.start_case(LUCIA, "Refund my returned blanket.", case_id=CASE)

    results = tool_results(await state_of(service))
    assert '"proposed":false' in results["p1"].content and "DISPUTE_OPEN" in results["p1"].content
    assert done.status == "completed"  # escalation is not gated
    assert [(e["type"], e["queue"]) for e in app.state.store.ledger] == [("escalation.created", "disputes")]
    assert [r["kind"] for r in done.receipts] == ["escalation"]


async def test_forged_proposal_id_cannot_move_money_even_if_approved(client, app):
    service = make_service(client, [
        AIMessage("", tool_calls=[execute("e1", "RP-0000000000", 25000)]),
        AIMessage("No refund was issued."),
    ])
    paused = await service.start_case(MAYA, "Refund me $250.", case_id=CASE)
    assert "does not exist in this case" in paused.approvals[0].description

    await service.submit_decisions(CASE, REVIEWER, [{"type": "approve"}])
    assert "UNKNOWN_PROPOSAL" in tool_results(await state_of(service))["e1"].content
    assert app.state.store.ledger == []


async def test_a_proposal_executes_at_most_once(client, app):
    service = make_service(client, [
        AIMessage("", tool_calls=[propose("p1", "ORD-240117", "duplicate_charge", 47626)]),
        AIMessage("", tool_calls=[execute("e1", DUP, 47626)]),
        AIMessage("", tool_calls=[execute("e2", DUP, 47626)]),  # the model tries again
        AIMessage("Refund RF-90001 was issued."),
    ])
    await service.start_case(MAYA, "Charged twice.", case_id=CASE)
    await service.submit_decisions(CASE, REVIEWER, [{"type": "approve"}])
    await service.submit_decisions(CASE, REVIEWER, [{"type": "approve"}])  # approved twice

    assert '"alreadyExecuted":true' in tool_results(await state_of(service))["e2"].content
    assert len(app.state.store.ledger) == 1


async def test_service_guards(client):
    service = make_service(client, [
        AIMessage("", tool_calls=[propose("p1", "ORD-240117", "duplicate_charge", 47626)]),
        AIMessage("", tool_calls=[execute("e1", DUP, 47626)]),
        AIMessage("Done."),
    ])
    await service.start_case(MAYA, "Charged twice.", case_id=CASE)

    with pytest.raises(CaseError):
        await service.submit_decisions("case-unknown", REVIEWER, [{"type": "approve"}])
    with pytest.raises(CaseError):
        await service.submit_decisions(CASE, "", [{"type": "approve"}])
    with pytest.raises(CaseError):
        await service.submit_decisions(CASE, REVIEWER, [])
    with pytest.raises(CaseError):
        await service.submit_decisions(CASE, REVIEWER, [{"type": "edit", "amount_cents": "20"}])
    with pytest.raises(CaseError):
        await service.start_case(MAYA, "again", case_id=CASE)

    done = await service.submit_decisions(CASE, REVIEWER, [{"type": "approve"}])
    with pytest.raises(CaseError):
        await service.submit_decisions(CASE, REVIEWER, [{"type": "approve"}])
    assert done.status == "completed"
