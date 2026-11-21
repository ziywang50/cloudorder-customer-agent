"""Customer scoping, data minimization, input validation, and retry rules of the
deep-agent read tools.

Tools run through a real create_agent tool node so ToolRuntime injection is exercised
exactly the way the deep agent uses it.
"""

import json

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from src.deep_agent.backend import BackendError, CloudOrderClient
from src.deep_agent.context import CaseContext
from src.deep_agent.tools import build_read_tools
from src.mock_backend.app import create_app
from tests.scripted import ScriptedChatModel, call

MAYA, ETHAN, GRACE = "CUS-10482", "CUS-10603", "CUS-10655"
TOOL_NAMES = ["get_customer_overview", "get_order", "get_payment_ledger", "get_refund_history",
              "get_shipment_tracking", "get_support_tickets", "check_refund_eligibility"]


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def store(app):
    return app.state.store


@pytest.fixture
async def tools(app):
    client = CloudOrderClient.in_process(app)
    yield build_read_tools(client)
    await client.aclose()


async def run_tools(tools, customer_id: str, calls: list[tuple[str, dict]]) -> list[dict]:
    """Run tool calls as `customer_id` and return each parsed result, in call order."""
    tool_calls = [call(name, args, f"call-{i}") for i, (name, args) in enumerate(calls)]
    model = ScriptedChatModel(scripts={"default": [AIMessage("", tool_calls=tool_calls),
                                                   AIMessage("done")]})
    agent = create_agent(model, tools=list(tools.values()), context_schema=CaseContext)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "check"}]},
        context=CaseContext(customer_id=customer_id, case_id="case-test"),
    )
    by_id = {m.tool_call_id: m.content for m in result["messages"] if m.type == "tool"}
    return [json.loads(by_id[f"call-{i}"]) for i in range(len(calls))]


class FlakyTransport(httpx.AsyncBaseTransport):
    """Fails the first `failures` requests with a connection error, then delegates."""

    def __init__(self, inner: httpx.AsyncBaseTransport, failures: int):
        self.inner = inner
        self.failures = failures
        self.calls = 0

    async def handle_async_request(self, request):
        self.calls += 1
        if self.calls <= self.failures:
            raise httpx.ConnectError("simulated outage", request=request)
        return await self.inner.handle_async_request(request)


def flaky_client(app, failures: int) -> tuple[CloudOrderClient, FlakyTransport]:
    transport = FlakyTransport(httpx.ASGITransport(app=app), failures)
    http = httpx.AsyncClient(transport=transport, base_url="http://cloudorder.mock")
    return CloudOrderClient(http, backoff_s=0.01), transport


# ---- customer scoping ----------------------------------------------------------------

@pytest.mark.parametrize("name", TOOL_NAMES)
def test_tool_schemas_never_take_a_customer(tools, name):
    properties = set(tools[name].tool_call_schema.model_json_schema().get("properties", {}))
    assert "runtime" not in properties
    assert not any("customer" in prop for prop in properties)


async def test_another_customers_order_reads_as_not_found(tools, store):
    log_start = len(store.access_log)
    results = await run_tools(tools, GRACE, [
        ("get_order", {"order_id": "ORD-240117"}),
        ("get_payment_ledger", {"order_id": "ORD-240117"}),
    ])
    assert [r["error"]["code"] for r in results] == ["ORDER_NOT_FOUND", "ORDER_NOT_FOUND"]
    assert "Nimbus" not in json.dumps(results)
    denials = [e for e in store.access_log[log_start:] if e["decision"] == "denied_not_owner"]
    assert len(denials) == 2
    assert all(e["onBehalfOf"] == GRACE for e in denials)


async def test_support_tickets_are_returned_for_the_owner(tools):
    [result] = await run_tools(tools, ETHAN, [("get_support_tickets", {})])
    assert [t["ticketId"] for t in result["tickets"]] == ["TCK-88213"]
    assert "pre-approved" in result["tickets"][0]["messages"][0]["body"]


# ---- data minimization ---------------------------------------------------------------

async def test_overview_and_order_views_omit_contact_details(tools):
    overview, order, ledger, tracking = await run_tools(tools, MAYA, [
        ("get_customer_overview", {}),
        ("get_order", {"order_id": "ORD-240117"}),
        ("get_payment_ledger", {"order_id": "ORD-240117"}),
        ("get_shipment_tracking", {"order_id": "ORD-240117"}),
    ])
    assert overview["firstName"] == "Maya"
    assert [o["orderId"] for o in overview["orders"]] == ["ORD-240117", "ORD-238544"]
    assert order["shipTo"] == "Austin, TX"
    assert order["paymentMethod"] == "visa ending 4417"
    blob = json.dumps([overview, order, ledger, tracking])
    for contact_detail in ("maya.chen@example.com", "555-0143", "4120 Duval", "Maya Chen", "78751"):
        assert contact_detail not in blob


async def test_ledger_tracking_and_eligibility_carry_what_investigators_need(tools):
    ledger, tracking, eligibility = await run_tools(tools, MAYA, [
        ("get_payment_ledger", {"order_id": "ORD-240117"}),
        ("get_shipment_tracking", {"order_id": "ORD-240117"}),
        ("check_refund_eligibility",
         {"order_id": "ORD-240117", "reason_code": "lost_package", "amount_cents": 1}),
    ])
    assert sum(t["type"] == "CAPTURE" for t in ledger["transactions"]) == 2
    assert all("note" in t for t in ledger["transactions"] if t["type"] == "CAPTURE")
    stalled = tracking["shipments"][1]
    assert stalled["shipmentId"] == "SHP-77121"
    assert stalled["lastScanAt"] and tracking["asOf"]
    assert eligibility["maxForReasonCents"] == 11905


# ---- input validation ----------------------------------------------------------------

@pytest.mark.parametrize("order_id", ["ORD-240117/../../api/admin/reset", "240117", "ORD-24O117", ""])
async def test_malformed_order_ids_never_reach_the_backend(tools, store, order_id):
    log_start = len(store.access_log)
    [result] = await run_tools(tools, MAYA, [("get_order", {"order_id": order_id})])
    assert result["error"]["code"] == "INVALID_ORDER_ID"
    assert len(store.access_log) == log_start


async def test_malformed_line_ids_are_rejected(tools):
    [result] = await run_tools(tools, MAYA, [(
        "check_refund_eligibility",
        {"order_id": "ORD-240117", "reason_code": "damaged_item_keep", "amount_cents": 100,
         "line_ids": ["L1"]},
    )])
    assert result["error"]["code"] == "INVALID_LINE_ID"


# ---- retries -------------------------------------------------------------------------

async def test_get_is_retried_through_transient_transport_failures(app):
    client, transport = flaky_client(app, failures=2)
    order = await client.get_order(MAYA, "ORD-240117")
    assert order["orderId"] == "ORD-240117"
    assert transport.calls == 3


async def test_persistent_outage_becomes_a_backend_unavailable_tool_error(app):
    client, transport = flaky_client(app, failures=99)
    [result] = await run_tools(build_read_tools(client), MAYA,
                               [("get_order", {"order_id": "ORD-240117"})])
    assert result["error"]["code"] == "BACKEND_UNAVAILABLE"
    assert transport.calls == 3


async def test_post_without_idempotency_key_is_not_retried(app):
    client, transport = flaky_client(app, failures=99)
    with pytest.raises(BackendError):
        await client._request("POST", "/api/refunds", customer_id=MAYA, json={})
    assert transport.calls == 1


async def test_post_with_idempotency_key_is_retried(app):
    client, transport = flaky_client(app, failures=99)
    with pytest.raises(BackendError):
        await client._request("POST", "/api/refunds", customer_id=MAYA, json={},
                              idempotency_key="k1")
    assert transport.calls == 3
