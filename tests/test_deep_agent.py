"""Wiring of the case-resolution deep agent, run end to end with a scripted model.

The script plays the coordinator and all three sub-agents on the flagship scenario
(duplicate charge plus lost package). It checks the deep-agent features the design relies
on: planning, parallel delegation, least-privilege sub-agent tools, case files, and the
read-only policy library. No LLM is called.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.deep_agent.agent import POLICIES_DIR, build_case_agent
from src.deep_agent.backend import CloudOrderClient
from src.deep_agent.context import CaseContext
from src.mock_backend.app import create_app
from tests.scripted import ScriptedChatModel, call

MAYA = "CUS-10482"
MONEY_TOOLS = {"get_payment_ledger", "get_shipment_tracking", "get_refund_history",
               "check_refund_eligibility"}
ROUTES = {
    "coordinator": "You are the CloudOrder case coordinator",
    "billing": "You are the billing investigator",
    "fulfillment": "You are the fulfillment investigator",
    "policy": "You are the policy analyst",
}


def flagship_script() -> dict[str, list[AIMessage]]:
    return {
        "coordinator": [
            AIMessage("", tool_calls=[call("write_todos", {"todos": [
                {"content": "Investigate the duplicate charge on ORD-240117", "status": "in_progress"},
                {"content": "Investigate the missing box on ORD-240117", "status": "in_progress"},
                {"content": "Confirm remedies with the policy analyst", "status": "pending"},
                {"content": "Reply to the customer", "status": "pending"},
            ]}, "c1")]),
            AIMessage("", tool_calls=[
                call("task", {"subagent_type": "billing-investigator",
                              "description": "ORD-240117: is there a duplicate charge?"}, "c2"),
                call("task", {"subagent_type": "fulfillment-investigator",
                              "description": "ORD-240117: which box is missing and is it lost?"}, "c3"),
            ]),
            AIMessage("", tool_calls=[call("task", {
                "subagent_type": "policy-analyst",
                "description": "ORD-240117: duplicate capture of 47626 cents; SHP-77121 lost. "
                               "Which remedies are allowed?",
            }, "c4")]),
            AIMessage("", tool_calls=[call("read_file", {"file_path": "/case/evidence/policy.md"}, "c5")]),
            AIMessage("You were charged twice and one box is lost. A specialist will review "
                      "refunds of $476.26 and $119.05."),
        ],
        "billing": [
            AIMessage("", tool_calls=[
                call("get_payment_ledger", {"order_id": "ORD-240117"}, "b1"),
                call("get_shipment_tracking", {"order_id": "ORD-240117"}, "b2"),  # not a billing tool
                call("execute_refund", {"proposal_id": "RP-x", "amount_cents": 47626}, "b2x"),  # coordinator only
            ]),
            AIMessage("", tool_calls=[call("write_file", {
                "file_path": "/case/evidence/billing.md",
                "content": "# Billing findings: ORD-240117\n## Conclusion\nDuplicate capture of 47626.",
            }, "b3")]),
            AIMessage("Duplicate capture of 47626 cents. Evidence: /case/evidence/billing.md"),
        ],
        "fulfillment": [
            AIMessage("", tool_calls=[call("get_shipment_tracking", {"order_id": "ORD-240117"}, "f1")]),
            AIMessage("", tool_calls=[call("write_file", {
                "file_path": "/case/evidence/fulfillment.md",
                "content": "# Fulfillment findings: ORD-240117\n## Conclusion\nSHP-77121 is lost.",
            }, "f2")]),
            AIMessage("SHP-77121 (L2, L3) meets the lost rule. Evidence: /case/evidence/fulfillment.md"),
        ],
        "policy": [
            AIMessage("", tool_calls=[
                call("read_file", {"file_path": "/policies/refunds.md"}, "p1"),
                call("write_file", {"file_path": "/policies/refunds.md", "content": "anything goes"}, "p2"),
            ]),
            AIMessage("", tool_calls=[
                call("check_refund_eligibility", {"order_id": "ORD-240117",
                                                  "reason_code": "duplicate_charge", "amount_cents": 1}, "p3"),
                call("check_refund_eligibility", {"order_id": "ORD-240117",
                                                  "reason_code": "lost_package", "amount_cents": 1}, "p4"),
            ]),
            AIMessage("", tool_calls=[call("write_file", {
                "file_path": "/case/evidence/policy.md",
                "content": "# Policy analysis: ORD-240117\n- duplicate_charge 47626 eligible\n"
                           "- lost_package 11905 eligible",
            }, "p5")]),
            AIMessage("Eligible: duplicate_charge 47626, lost_package 11905."),
        ],
    }


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    backend = CloudOrderClient.in_process(app)
    yield backend
    await backend.aclose()


@pytest.fixture
def model():
    return ScriptedChatModel(scripts=flagship_script(), routes=ROUTES)


@pytest.fixture
def agent(model, client):
    return build_case_agent(model=model, subagent_model=model, client=client)


async def run_case(agent, message: str) -> tuple[dict, dict[str, ToolMessage]]:
    """Run one case, returning the final state and every ToolMessage across all agents."""
    config = {"configurable": {"thread_id": "case-test"}}
    tool_messages: dict[str, ToolMessage] = {}
    async for part in agent.astream(
        {"messages": [{"role": "user", "content": message}]}, config,
        context=CaseContext(customer_id=MAYA, case_id="case-test"),
        stream_mode="updates", subgraphs=True, version="v2",
    ):
        for update in (part["data"] or {}).values():
            for item in update if isinstance(update, list) else [update]:
                for message_ in (item or {}).get("messages", []) if isinstance(item, dict) else []:
                    if isinstance(message_, ToolMessage):
                        tool_messages[message_.tool_call_id] = message_
    state = await agent.aget_state(config)
    return state.values, tool_messages


def test_task_tool_offers_exactly_the_three_investigators(agent):
    description = agent.nodes["tools"].bound.tools_by_name["task"].description
    for name in ("billing-investigator", "fulfillment-investigator", "policy-analyst"):
        assert f"- {name}:" in description
    assert "- general-purpose:" not in description


def test_coordinator_has_no_payment_or_shipping_tools(agent):
    names = set(agent.nodes["tools"].bound.tools_by_name)
    assert {"task", "write_todos", "read_file", "get_customer_overview"} <= names
    assert not names & MONEY_TOOLS


async def test_flagship_case_plans_delegates_and_files_evidence(agent, model, app):
    policy_text = (POLICIES_DIR / "refunds.md").read_text(encoding="utf-8")
    state, tools = await run_case(agent, "I was charged twice and one box never arrived.")

    # Planning: the coordinator's todo list is in the checkpointed state.
    assert [t["content"] for t in state["todos"]][-1] == "Reply to the customer"

    # Delegation: each investigator ran, billing and fulfillment from one coordinator turn.
    assert model.calls.count("billing") == 3
    assert model.calls.count("fulfillment") == 3
    assert model.calls.count("policy") == 4
    assert "47626" in tools["c2"].content and "SHP-77121" in tools["c3"].content

    # Least privilege: the billing investigator cannot reach shipment tracking or money tools.
    assert tools["b2"].status == "error"
    assert "get_shipment_tracking" in tools["b2"].content
    assert tools["b2x"].status == "error"
    assert "execute_refund" in tools["b2x"].content

    # Case files: every investigator's evidence is in the case filesystem.
    assert {"/case/evidence/billing.md", "/case/evidence/fulfillment.md",
            "/case/evidence/policy.md"} <= set(state["files"])
    assert "11905" in state["files"]["/case/evidence/policy.md"]["content"]
    assert "lost_package" in tools["c5"].content

    # Policies are readable but not writable, on disk or in the case filesystem.
    assert "lost_package" in tools["p1"].content
    assert "permission denied" in tools["p2"].content.lower()
    assert (POLICIES_DIR / "refunds.md").read_text(encoding="utf-8") == policy_text
    assert "/policies/refunds.md" not in state["files"]

    # Eligibility came from the backend rules engine, as the authenticated customer.
    assert '"maxForReasonCents":47626' in tools["p3"].content
    assert '"maxForReasonCents":11905' in tools["p4"].content
    store = app.state.store
    assert {e["onBehalfOf"] for e in store.access_log} == {MAYA}
    assert all(e["decision"] == "allowed" for e in store.access_log)
    assert store.ledger == []  # the script never proposed or executed a refund

    assert state["messages"][-1].content.startswith("You were charged twice")
