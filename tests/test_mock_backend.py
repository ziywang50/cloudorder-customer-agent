"""Money rules, access control, and scenario consistency of the mock CloudOrder backend.

Every evaluation number for the deep-agent path depends on these rules, so they are
tested through the same HTTP API the agent's tools call. The data is anchored to a fixed
AS_OF instant, so none of these assertions drift with the wall clock.
"""

import asyncio
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from src.mock_backend.app import create_app
from src.mock_backend.data import INJECTED_TICKET_BODY
from src.mock_backend.db import UniqueViolation
from src.mock_backend.scenarios import SCENARIOS, SCENARIOS_BY_ID
from src.mock_backend.store import IDEMPOTENCY_TTL, CloudOrderStore, StoreError

MAYA, DANIEL, LUCIA, ETHAN, GRACE, ARJUN = (
    "CUS-10482", "CUS-10519", "CUS-10577", "CUS-10603", "CUS-10655", "CUS-10702"
)
REVIEWER = "reviewer@cloudorder.example"

# Every backend test runs on SQLite. Point this variable at a disposable Postgres database
# to run the same tests there too; the store drops and re-seeds its tables.
POSTGRES_TEST_URL_ENV = "CLOUDORDER_MOCK_TEST_POSTGRES_URL"
BACKENDS = ["sqlite"] + (["postgres"] if os.environ.get(POSTGRES_TEST_URL_ENV) else [])


@pytest.fixture(params=BACKENDS)
def store(request, tmp_path):
    if request.param == "postgres":
        return CloudOrderStore(database_url=os.environ[POSTGRES_TEST_URL_ENV])
    return CloudOrderStore(database_url=f"sqlite:///{(tmp_path / 'cloudorder.db').as_posix()}")


@pytest.fixture
async def client(store):
    transport = httpx.ASGITransport(app=create_app(store))
    async with httpx.AsyncClient(transport=transport, base_url="http://cloudorder.mock") as http:
        yield http


def as_customer(customer_id: str) -> dict[str, str]:
    return {"X-On-Behalf-Of": customer_id}


def codes(result: dict) -> list[str]:
    return [blocker["code"] for blocker in result["blockingReasons"]]


async def preview(client, customer_id: str, **body) -> dict:
    response = await client.post("/api/refunds/preview", headers=as_customer(customer_id), json=body)
    assert response.status_code == 200, response.text
    return response.json()


async def create_refund(client, customer_id: str, key: str, **body) -> httpx.Response:
    payload = {"caseId": "case-test", "approvedBy": REVIEWER, **body}
    headers = {**as_customer(customer_id), "Idempotency-Key": key}
    return await client.post("/api/refunds", headers=headers, json=payload)


# ---- scenario data consistency -------------------------------------------------------

CUSTOMER_NAMES = {
    MAYA: "Maya Chen",
    DANIEL: "Daniel Okafor",
    LUCIA: "Lucía Fernández",
    ETHAN: "Ethan Brooks",
    GRACE: "Grace Liu",
    ARJUN: "Arjun Patel",
}

# scenario id -> (order the customer talks about, words that must appear in both the
# customer's message and that order's product names)
PRODUCT_MENTIONS = {
    "duplicate-charge-lost-package": ("ORD-240117", ["monitor", "mouse", "hub"]),
    "auth-hold-not-duplicate": ("ORD-240233", ["backpack"]),
    "open-chargeback-escalation": ("ORD-239884", ["blanket"]),
    "ticket-prompt-injection": ("ORD-240301", ["kettle"]),
    "reviewer-edits-amount": ("ORD-240156", ["keyboard"]),
}


def test_scenario_customers_exist_with_expected_names(store):
    assert len(SCENARIOS) == 6
    assert {s.customer_id for s in SCENARIOS} == set(CUSTOMER_NAMES)
    for customer_id, name in CUSTOMER_NAMES.items():
        assert store.customers[customer_id]["name"] == name


@pytest.mark.parametrize("scenario_id", sorted(PRODUCT_MENTIONS))
def test_products_in_message_match_the_order(store, scenario_id):
    scenario = SCENARIOS_BY_ID[scenario_id]
    order_id, words = PRODUCT_MENTIONS[scenario_id]
    order = store.orders[order_id]
    assert order["customerId"] == scenario.customer_id
    product_names = " ".join(item["productName"] for item in order["items"]).lower()
    for word in words:
        assert word in scenario.message.lower(), f"{word!r} missing from the message"
        assert word in product_names, f"{word!r} missing from {order_id}'s products"


def test_order_ids_in_messages_belong_to_the_right_customer(store):
    for scenario in SCENARIOS:
        for order_id in re.findall(r"ORD-\d{6}", scenario.message):
            owner = store.orders[order_id]["customerId"]
            if scenario.id == "cross-customer-order":
                assert owner != scenario.customer_id
            else:
                assert owner == scenario.customer_id


@pytest.mark.parametrize(
    ("scenario_id", "order_id"),
    [
        ("duplicate-charge-lost-package", "ORD-240117"),
        ("auth-hold-not-duplicate", "ORD-240233"),
        ("open-chargeback-escalation", "ORD-239884"),
    ],
)
def test_dollar_amounts_quoted_by_customers_equal_order_totals(store, scenario_id, order_id):
    quoted = re.findall(r"\$(\d+\.\d{2})", SCENARIOS_BY_ID[scenario_id].message)
    assert quoted, "the scenario message should quote an amount"
    assert all(round(float(q) * 100) == store.orders[order_id]["totalCents"] for q in quoted)


async def test_expected_refunds_match_backend_limits(client, store):
    for scenario in SCENARIOS:
        for expected in scenario.expected_refunds:
            assert store.orders[expected.order_id]["customerId"] == scenario.customer_id
            result = await preview(
                client, scenario.customer_id, orderId=expected.order_id,
                amountCents=expected.amount_cents, reasonCode=expected.reason_code,
                lineIds=list(expected.line_ids),
            )
            assert result["eligible"], (scenario.id, result["blockingReasons"])
            if scenario.reviewer_edit_amount_cents is None:
                # The full allowed amount is the correct remedy.
                assert result["maxForReasonCents"] == expected.amount_cents
            else:
                assert expected.amount_cents == scenario.reviewer_edit_amount_cents
                assert expected.amount_cents < result["maxForReasonCents"]


def test_injection_ticket_is_on_the_scenario_customer(store):
    tickets = store.tickets[ETHAN]
    assert [t["ticketId"] for t in tickets] == ["TCK-88213"]
    assert tickets[0]["orderId"] == "ORD-240301"
    assert tickets[0]["messages"][0]["body"] == INJECTED_TICKET_BODY


def test_leak_terms_are_real_facts_of_the_foreign_order(store):
    """The cross-customer leak check is only meaningful if every term is Maya's data."""
    scenario = SCENARIOS_BY_ID["cross-customer-order"]
    lost_package = store.preview_refund(MAYA, order_id="ORD-240117", amount_cents=1,
                                        reason_code="lost_package", line_ids=[])
    foreign = " ".join(
        str(part) for part in (
            store.customers[MAYA],
            store.orders["ORD-240117"],
            store.transactions["ORD-240117"],
            store.shipments["ORD-240117"],
            lost_package["basis"],  # derived facts such as the $119.05 refund value
        )
    )
    for term in scenario.must_not_mention:
        assert term in foreign, term
        assert term not in str(store.orders["ORD-240089"]), term  # not also Grace's own data


# ---- refund limits per reason ----------------------------------------------------------

async def test_duplicate_charge_limit_is_the_extra_capture(client):
    result = await preview(client, MAYA, orderId="ORD-240117", amountCents=47626,
                           reasonCode="duplicate_charge")
    assert result["eligible"]
    assert result["maxForReasonCents"] == 47626
    assert result["capturedCents"] == 2 * 47626

    over = await preview(client, MAYA, orderId="ORD-240117", amountCents=47627,
                         reasonCode="duplicate_charge")
    assert codes(over) == ["EXCEEDS_REASON_MAX"]


async def test_voided_authorization_is_not_a_duplicate(client):
    result = await preview(client, DANIEL, orderId="ORD-240233", amountCents=8807,
                           reasonCode="duplicate_charge")
    assert not result["eligible"]
    assert codes(result) == ["NO_DUPLICATE_CAPTURE"]
    assert result["maxForReasonCents"] == 0


async def test_lost_package_limit_covers_only_the_lost_shipment(client):
    result = await preview(client, MAYA, orderId="ORD-240117", amountCents=11905,
                           reasonCode="lost_package")
    assert result["eligible"]
    assert result["maxForReasonCents"] == 11905  # mouse + hub with tax, not the monitor
    assert "SHP-77121" in result["basis"]


async def test_lost_package_rejected_when_every_shipment_was_delivered(client):
    result = await preview(client, ARJUN, orderId="ORD-240156", amountCents=1000,
                           reasonCode="lost_package")
    assert codes(result) == ["NOT_LOST"]


async def test_damaged_item_keep_is_capped_at_25_percent_of_the_line(client):
    ok = await preview(client, ARJUN, orderId="ORD-240156", amountCents=3538,
                       reasonCode="damaged_item_keep", lineIds=["ORD-240156-L1"])
    assert ok["eligible"]
    assert ok["maxForReasonCents"] == 3538  # floor(25% of $141.53)

    over = await preview(client, ARJUN, orderId="ORD-240156", amountCents=3539,
                         reasonCode="damaged_item_keep", lineIds=["ORD-240156-L1"])
    assert codes(over) == ["EXCEEDS_REASON_MAX"]


@pytest.mark.parametrize(
    ("line_ids", "code"),
    [([], "LINE_IDS_REQUIRED"), (["ORD-240156-L9"], "UNKNOWN_LINE")],
)
async def test_damaged_item_keep_needs_valid_lines(client, line_ids, code):
    result = await preview(client, ARJUN, orderId="ORD-240156", amountCents=1000,
                           reasonCode="damaged_item_keep", lineIds=line_ids)
    assert codes(result) == [code]


async def test_amount_can_never_exceed_what_is_still_refundable(client):
    # The injected "$250 loyalty refund" on a $69.64 kettle order.
    result = await preview(client, ETHAN, orderId="ORD-240301", amountCents=25000,
                           reasonCode="damaged_item_keep", lineIds=["ORD-240301-L1"])
    assert not result["eligible"]
    assert "EXCEEDS_REFUNDABLE" in codes(result)


@pytest.mark.parametrize(
    ("amount_cents", "reason_code", "code"),
    [(0, "duplicate_charge", "INVALID_AMOUNT"), (500, "goodwill", "UNKNOWN_REASON")],
)
async def test_invalid_amounts_and_reasons_are_blocked(client, amount_cents, reason_code, code):
    result = await preview(client, MAYA, orderId="ORD-240117", amountCents=amount_cents,
                           reasonCode=reason_code)
    assert code in codes(result)


async def test_limits_shrink_after_refunds_and_reasons_stay_independent(client):
    assert (await create_refund(client, MAYA, "dup-1", orderId="ORD-240117", amountCents=47626,
                                reasonCode="duplicate_charge")).status_code == 201
    again = await preview(client, MAYA, orderId="ORD-240117", amountCents=1,
                          reasonCode="duplicate_charge")
    assert codes(again) == ["NO_DUPLICATE_CAPTURE"]

    # A duplicate-charge refund does not shrink the lost-package allowance.
    assert (await create_refund(client, MAYA, "lost-1", orderId="ORD-240117", amountCents=5000,
                                reasonCode="lost_package")).status_code == 201
    remaining = await preview(client, MAYA, orderId="ORD-240117", amountCents=6905,
                              reasonCode="lost_package")
    assert remaining["eligible"]
    assert remaining["maxForReasonCents"] == 6905


# ---- open chargeback -------------------------------------------------------------------

@pytest.mark.parametrize("reason_code", ["duplicate_charge", "lost_package", "damaged_item_keep"])
async def test_open_chargeback_blocks_every_refund_reason(client, reason_code):
    result = await preview(client, LUCIA, orderId="ORD-239884", amountCents=13120,
                           reasonCode=reason_code, lineIds=["ORD-239884-L1"])
    assert "DISPUTE_OPEN" in codes(result)


async def test_open_chargeback_rejects_refund_creation(client, store):
    response = await create_refund(client, LUCIA, "cb-1", orderId="ORD-239884", amountCents=3000,
                                   reasonCode="damaged_item_keep", lineIds=["ORD-239884-L1"])
    assert response.status_code == 409
    assert response.json()["errorCode"] == "REFUND_BLOCKED"
    assert "DISPUTE_OPEN" in [b["code"] for b in response.json()["blockingReasons"]]
    assert store.ledger == []


async def test_disputes_escalation_is_recorded(client, store):
    headers = {**as_customer(LUCIA), "Idempotency-Key": "esc-1"}
    body = {"queue": "disputes", "priority": "high", "caseId": "case-test",
            "orderId": "ORD-239884", "summary": "Chargeback dp_1QK2ZrT9 open; refund frozen."}
    first = await client.post("/api/escalations", headers=headers, json=body)
    replay = await client.post("/api/escalations", headers=headers, json=body)
    assert first.status_code == replay.status_code == 201
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert replay.json()["escalationId"] == first.json()["escalationId"]
    assert [e["type"] for e in store.ledger] == ["escalation.created"]
    assert store.ledger[0]["queue"] == "disputes"


# ---- idempotency and approval ------------------------------------------------------------

DUPLICATE_REFUND = {"orderId": "ORD-240117", "amountCents": 47626, "reasonCode": "duplicate_charge"}


async def test_refund_creation_records_ledger_and_payment(client, store):
    response = await create_refund(client, MAYA, "k1", **DUPLICATE_REFUND)
    assert response.status_code == 201
    refund = response.json()
    assert refund["status"] == "PENDING"
    assert refund["approvedBy"] == REVIEWER
    assert store.ledger == [{
        "seq": 1, "type": "refund.created", "at": refund["createdAt"],
        "refundId": refund["refundId"], "orderId": "ORD-240117", "customerId": MAYA,
        "amountCents": 47626, "reasonCode": "duplicate_charge", "approvedBy": REVIEWER,
        "caseId": "case-test", "idempotencyKey": "k1",
    }]
    refund_txns = [t for t in store.transactions["ORD-240117"] if t["type"] == "REFUND"]
    assert [(t["refundId"], t["amountCents"]) for t in refund_txns] == [(refund["refundId"], 47626)]


async def test_same_idempotency_key_replays_without_a_second_refund(client, store):
    first = await create_refund(client, MAYA, "k1", **DUPLICATE_REFUND)
    replay = await create_refund(client, MAYA, "k1", **DUPLICATE_REFUND)
    assert first.status_code == replay.status_code == 201  # the original status, not 200
    assert "Idempotent-Replayed" not in first.headers
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert replay.json() == first.json()
    assert len(store.ledger) == 1


async def test_reused_key_with_a_different_payload_is_rejected(client, store):
    await create_refund(client, MAYA, "k1", **DUPLICATE_REFUND)
    response = await create_refund(client, MAYA, "k1", **{**DUPLICATE_REFUND, "amountCents": 100})
    assert response.status_code == 409
    assert response.json()["errorCode"] == "IDEMPOTENCY_KEY_REUSED"
    assert len(store.ledger) == 1


async def test_new_key_cannot_refund_the_duplicate_twice(client, store):
    await create_refund(client, MAYA, "k1", **DUPLICATE_REFUND)
    response = await create_refund(client, MAYA, "k2", **DUPLICATE_REFUND)
    assert response.status_code == 409
    assert "NO_DUPLICATE_CAPTURE" in [b["code"] for b in response.json()["blockingReasons"]]
    assert len(store.ledger) == 1


async def test_mutations_require_an_idempotency_key(client):
    response = await client.post(
        "/api/refunds", headers=as_customer(MAYA),
        json={**DUPLICATE_REFUND, "caseId": "case-test", "approvedBy": REVIEWER},
    )
    assert response.status_code == 400
    assert response.json()["errorCode"] == "IDEMPOTENCY_KEY_REQUIRED"


async def test_refunds_must_name_an_approver(client, store):
    response = await create_refund(client, MAYA, "k1", **{**DUPLICATE_REFUND, "approvedBy": ""})
    assert response.status_code == 422
    with pytest.raises(StoreError) as error:
        store.create_refund(MAYA, idempotency_key="k2", order_id="ORD-240117", amount_cents=47626,
                            reason_code="duplicate_charge", line_ids=[], case_id="case-test",
                            approved_by="   ")
    assert error.value.code == "APPROVER_REQUIRED"
    assert store.ledger == []


# ---- access control ----------------------------------------------------------------------

@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/orders/ORD-240117"),
        ("GET", "/api/orders/ORD-240117/payments"),
        ("GET", "/api/orders/ORD-240117/shipments"),
        ("GET", "/api/orders/ORD-240117/refunds"),
        ("GET", "/api/customers/CUS-10482"),
        ("GET", "/api/customers/CUS-10482/orders"),
        ("GET", "/api/customers/CUS-10482/tickets"),
    ],
)
async def test_another_customers_resources_return_404(client, store, method, path):
    response = await client.request(method, path, headers=as_customer(GRACE))
    assert response.status_code == 404
    assert store.access_log[-1]["decision"] == "denied_not_owner"
    assert store.access_log[-1]["onBehalfOf"] == GRACE


async def test_foreign_order_looks_exactly_like_a_missing_order(client):
    foreign = await client.get("/api/orders/ORD-240117", headers=as_customer(GRACE))
    missing = await client.get("/api/orders/ORD-999999", headers=as_customer(GRACE))
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json()["errorCode"] == missing.json()["errorCode"] == "ORDER_NOT_FOUND"
    assert foreign.json()["message"].replace("ORD-240117", "X") == \
        missing.json()["message"].replace("ORD-999999", "X")


async def test_cross_customer_refund_preview_and_creation_are_404(client, store):
    preview_response = await client.post(
        "/api/refunds/preview", headers=as_customer(GRACE),
        json={"orderId": "ORD-240117", "amountCents": 11905, "reasonCode": "lost_package"},
    )
    create_response = await create_refund(client, GRACE, "x1", orderId="ORD-240117",
                                          amountCents=11905, reasonCode="lost_package")
    assert preview_response.status_code == 404
    assert create_response.status_code == 404
    assert store.ledger == []


async def test_customer_endpoints_require_on_behalf_of(client):
    response = await client.get("/api/orders/ORD-240117/payments")
    assert response.status_code == 401
    assert response.json()["errorCode"] == "ON_BEHALF_OF_REQUIRED"


async def test_legacy_order_contract_still_serves_the_existing_tools(client):
    order = (await client.get("/api/orders/ORD-240117")).json()
    for field in ("orderId", "status", "totalAmount", "buyerName", "buyerPhone", "buyerAddress",
                  "createdAt", "updatedAt"):
        assert field in order
    assert order["totalAmount"] == 476.26
    assert {"productName", "quantity", "productPrice", "subtotal"} <= set(order["items"][0])
    scoped = await client.get("/api/orders/ORD-240117", headers=as_customer(GRACE))
    assert scoped.status_code == 404


async def test_reset_restores_the_seed(client, store):
    await create_refund(client, MAYA, "k1", **DUPLICATE_REFUND)
    assert (await client.post("/api/admin/reset")).status_code == 200
    assert store.ledger == []
    assert store.access_log == []
    assert store.refunds["ORD-240117"] == []


# ---- production idempotency and transactions -----------------------------------------------

def refund_args(**overrides) -> dict:
    """Store-level arguments for Maya's duplicate-charge refund."""
    return {"idempotency_key": "k1", "order_id": "ORD-240117", "amount_cents": 47626,
            "reason_code": "duplicate_charge", "line_ids": [], "case_id": "case-test",
            "approved_by": REVIEWER, **overrides}


def idempotency_records(store) -> list[dict]:
    with store.db.transaction() as tx:
        return tx.fetchall("SELECT customer_id, endpoint, idempotency_key, status_code "
                           "FROM idempotency_records ORDER BY customer_id")


def run_concurrently(*calls):
    """Start every call at the same moment on its own thread; return results or StoreErrors."""
    barrier = threading.Barrier(len(calls))

    def run(fn):
        barrier.wait()
        try:
            return fn()
        except StoreError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return [future.result() for future in [pool.submit(run, fn) for fn in calls]]


def pause_at(monkeypatch, store, stage: str, seconds: float = 0.3) -> None:
    """Hold a write transaction open at `stage` so concurrent requests really overlap."""
    def checkpoint(current: str) -> None:
        if current == stage:
            time.sleep(seconds)

    monkeypatch.setattr(store, "_checkpoint", checkpoint)


async def test_keys_are_scoped_per_customer(client, store):
    maya = await create_refund(client, MAYA, "shared-key", **DUPLICATE_REFUND)
    arjun = await create_refund(client, ARJUN, "shared-key", orderId="ORD-240156",
                                amountCents=2000, reasonCode="damaged_item_keep",
                                lineIds=["ORD-240156-L1"])
    assert maya.status_code == arjun.status_code == 201
    assert "Idempotent-Replayed" not in arjun.headers
    assert [r["customer_id"] for r in idempotency_records(store)] == [MAYA, ARJUN]
    assert len(store.ledger) == 2


async def test_blocked_outcome_is_stored_and_replayed(client, store):
    body = {"orderId": "ORD-239884", "amountCents": 3000, "reasonCode": "damaged_item_keep",
            "lineIds": ["ORD-239884-L1"]}
    first = await create_refund(client, LUCIA, "cb-1", **body)
    replay = await create_refund(client, LUCIA, "cb-1", **body)
    assert first.status_code == replay.status_code == 409
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert replay.json() == first.json()
    assert [r["status_code"] for r in idempotency_records(store)] == [409]
    assert store.ledger == []


def test_validation_errors_do_not_consume_the_key(store):
    with pytest.raises(StoreError) as error:
        store.create_refund(MAYA, **refund_args(approved_by="  "))
    assert error.value.code == "APPROVER_REQUIRED"
    with pytest.raises(StoreError) as error:
        store.create_refund(GRACE, **refund_args())  # Maya's order, asked for by Grace
    assert error.value.status == 404
    assert idempotency_records(store) == []

    outcome = store.create_refund(MAYA, **refund_args())
    assert outcome.status_code == 201
    assert not outcome.replayed


@pytest.mark.parametrize("key", ["has space", "tab\tkey", "x" * 256])
async def test_malformed_idempotency_keys_are_rejected(client, store, key):
    response = await create_refund(client, MAYA, key, **DUPLICATE_REFUND)
    assert response.status_code == 400
    assert response.json()["errorCode"] == "INVALID_IDEMPOTENCY_KEY"
    assert store.ledger == []


def test_idempotency_records_expire_after_24_hours(store):
    first = store.create_refund(MAYA, **refund_args())
    store.advance_clock(int(IDEMPOTENCY_TTL.total_seconds()) + 1)
    # Before expiry this payload change would be IDEMPOTENCY_KEY_REUSED.
    reused = store.create_refund(MAYA, **refund_args(reason_code="lost_package", amount_cents=1000))
    assert first.status_code == reused.status_code == 201
    assert not reused.replayed
    assert len(store.refunds["ORD-240117"]) == 2


def test_unique_constraint_backs_the_idempotency_scope(store):
    sql = ("INSERT INTO idempotency_records (customer_id, endpoint, idempotency_key, request_hash, "
           "status_code, response_body, created_at, expires_at) VALUES (:customer_id, :endpoint, "
           ":key, :request_hash, :status_code, :response_body, :created_at, :expires_at)")
    row = {"customer_id": MAYA, "endpoint": "POST /api/refunds", "key": "k1", "request_hash": "h",
           "status_code": 201, "response_body": "{}", "created_at": "t", "expires_at": "t"}
    with pytest.raises(UniqueViolation):
        with store.db.transaction(write=True) as tx:
            tx.execute(sql, row)
            tx.execute(sql, row)
    assert idempotency_records(store) == []  # the first insert rolled back with the second


def test_concurrent_requests_with_the_same_key_create_one_refund(store, monkeypatch):
    pause_at(monkeypatch, store, "balance_checked")
    results = run_concurrently(lambda: store.create_refund(MAYA, **refund_args()),
                               lambda: store.create_refund(MAYA, **refund_args()))
    assert [r.status_code for r in results] == [201, 201]
    assert sorted(r.replayed for r in results) == [False, True]
    assert results[0].body == results[1].body
    assert len(store.refunds["ORD-240117"]) == 1
    assert len(store.ledger) == 1
    assert len(idempotency_records(store)) == 1


def test_concurrent_refunds_cannot_together_exceed_the_refundable_balance(store, monkeypatch):
    # A lost chargeback already took back $800.00 of the $952.52 captured, so only $152.52 is
    # refundable. Each $100.00 request fits on its own; together they would over-refund.
    store._seed_transactions["ORD-240117"].append(
        {"transactionId": "txn_cb_test", "type": "CHARGEBACK", "status": "LOST", "amountCents": 80000}
    )
    pause_at(monkeypatch, store, "balance_checked")
    results = run_concurrently(
        lambda: store.create_refund(MAYA, **refund_args(idempotency_key="a", amount_cents=10000)),
        lambda: store.create_refund(MAYA, **refund_args(idempotency_key="b", amount_cents=10000,
                                                        reason_code="lost_package")),
    )
    assert sorted(r.status_code for r in results) == [201, 409]
    blocked = next(r for r in results if r.status_code == 409)
    assert "EXCEEDS_REFUNDABLE" in [b["code"] for b in blocked.body["blockingReasons"]]
    assert sum(r["amountCents"] for r in store.refunds["ORD-240117"]) == 10000
    assert len(store.ledger) == 1


def test_concurrent_escalations_with_the_same_key_create_one(store, monkeypatch):
    pause_at(monkeypatch, store, "escalation_inserted")

    def escalate():
        return store.create_escalation(LUCIA, idempotency_key="esc-1", queue="disputes",
                                       priority="high", case_id="case-test",
                                       summary="Chargeback dp_1QK2ZrT9 open; refund frozen.")

    results = run_concurrently(escalate, escalate)
    assert sorted(r.replayed for r in results) == [False, True]
    assert len(store.escalations) == 1
    assert len(store.ledger) == 1


async def test_concurrent_http_retries_create_one_refund(client, store, monkeypatch):
    pause_at(monkeypatch, store, "balance_checked")
    first, second = await asyncio.gather(
        create_refund(client, MAYA, "k1", **DUPLICATE_REFUND),
        create_refund(client, MAYA, "k1", **DUPLICATE_REFUND),
    )
    assert first.status_code == second.status_code == 201
    replayed = sorted(r.headers.get("Idempotent-Replayed", "false") for r in (first, second))
    assert replayed == ["false", "true"]
    assert len(store.ledger) == 1


def test_crash_mid_transaction_saves_nothing(store, monkeypatch):
    clock_before = store.now()

    def crash(stage: str) -> None:
        if stage == "refund_inserted":
            raise RuntimeError("process died between the refund insert and the ledger insert")

    monkeypatch.setattr(store, "_checkpoint", crash)
    with pytest.raises(RuntimeError):
        store.create_refund(MAYA, **refund_args())
    assert store.refunds["ORD-240117"] == []
    assert store.ledger == []
    assert idempotency_records(store) == []
    assert store.now() == clock_before

    monkeypatch.undo()
    retry = store.create_refund(MAYA, **refund_args())
    assert retry.status_code == 201
    assert not retry.replayed  # nothing was stored, so the retry runs for real
    assert len(store.refunds["ORD-240117"]) == 1
