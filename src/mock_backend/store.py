"""CloudOrder mock state plus the money rules the real services enforce.

Seed facts (customers, orders, captures, chargebacks, shipments, tickets) are immutable and
held in memory. Everything the API can change (refunds, escalations, the ledger,
idempotency records, and the backend clock) lives in a database: SQLite by default, or
Postgres when CLOUDORDER_MOCK_DATABASE_URL is set in the process environment.

Each write is one transaction: the business row, its ledger entry, and its idempotency
record commit together or not at all. Refund writes lock the order before reading, so the
refundable balance is re-checked against committed refunds and two requests cannot both
spend it.

The agent never imports this module; it only talks to the HTTP API in app.py.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import weakref
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

from src.mock_backend.data import AS_OF, build_seed, iso, parse_iso
from src.mock_backend.db import Database, Transaction, UniqueViolation

DATABASE_URL_ENV = "CLOUDORDER_MOCK_DATABASE_URL"

REFUND_REASONS = ("duplicate_charge", "lost_package", "damaged_item_keep")
ESCALATION_SLA_HOURS = {"disputes": 48, "tier2_support": 24, "trust_safety": 72}
LOST_PACKAGE_MIN_DAYS_WITHOUT_SCAN = 5
DAMAGED_KEEP_MAX_FRACTION = Decimal("0.25")
REFUND_SETTLEMENT_DAYS = 7

IDEMPOTENCY_TTL = timedelta(hours=24)
IDEMPOTENCY_KEY_PATTERN = re.compile(r"[\x21-\x7e]{1,255}")  # visible ASCII, no spaces
REFUNDS_ENDPOINT = "POST /api/refunds"
ESCALATIONS_ENDPOINT = "POST /api/escalations"
_INITIAL_COUNTERS = {"clock": 0, "refund": 90000, "escalation": 50000}


class StoreError(Exception):
    """A business or access error that maps onto an HTTP status and error code."""

    def __init__(self, status: int, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details


@dataclass(frozen=True)
class Outcome:
    """What an idempotent write returns: the HTTP status and body, and whether it is a replay."""

    status_code: int
    body: dict[str, Any]
    replayed: bool = False


def _blocker(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _usd(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def _hash_request(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check_idempotency_key(key: str) -> None:
    if not IDEMPOTENCY_KEY_PATTERN.fullmatch(key or ""):
        raise StoreError(
            400, "INVALID_IDEMPOTENCY_KEY",
            "Idempotency-Key must be 1-255 visible ASCII characters with no spaces.",
        )


def _refund_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "refundId": row["refund_id"],
        "orderId": row["order_id"],
        "customerId": row["customer_id"],
        "amountCents": int(row["amount_cents"]),
        "currency": row["currency"],
        "reasonCode": row["reason_code"],
        "lineIds": json.loads(row["line_ids"]),
        "status": row["status"],
        "createdAt": row["created_at"],
        "expectedSettlementBy": row["expected_settlement_by"],
        "approvedBy": row["approved_by"],
        "caseId": row["case_id"],
        "note": row["note"],
    }


def _escalation_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "escalationId": row["escalation_id"],
        "customerId": row["customer_id"],
        "orderId": row["order_id"],
        "queue": row["queue"],
        "priority": row["priority"],
        "status": row["status"],
        "summary": row["summary"],
        "caseId": row["case_id"],
        "createdAt": row["created_at"],
        "firstResponseDueBy": row["first_response_due_by"],
    }


class _PerOrderView(Mapping[str, list[dict[str, Any]]]):
    """Read-only `store.refunds[order_id]` / `store.transactions[order_id]` backed by the DB."""

    def __init__(self, store: CloudOrderStore, loader: Callable[[Transaction, str], list[dict]]):
        self._store = store
        self._loader = loader

    def __getitem__(self, order_id: str) -> list[dict[str, Any]]:
        if order_id not in self._store.orders:
            raise KeyError(order_id)
        with self._store.db.transaction() as tx:
            return self._loader(tx, order_id)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store.orders)

    def __len__(self) -> int:
        return len(self._store.orders)


class CloudOrderStore:
    def __init__(self, database_url: str | None = None) -> None:
        url = database_url or os.environ.get(DATABASE_URL_ENV)
        if url is None:
            directory = tempfile.mkdtemp(prefix="cloudorder-mock-")
            weakref.finalize(self, shutil.rmtree, directory, ignore_errors=True)
            url = f"sqlite:///{Path(directory, 'cloudorder.db').as_posix()}"
        self.db = Database(url)
        self.db.create_schema()
        self._log_lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Restore the seed: in-memory facts and an empty, re-seeded database."""
        seed = build_seed()
        self.customers: dict[str, dict[str, Any]] = seed["customers"]
        self.orders: dict[str, dict[str, Any]] = seed["orders"]
        self._seed_transactions: dict[str, list[dict[str, Any]]] = seed["transactions"]
        self.shipments: dict[str, list[dict[str, Any]]] = seed["shipments"]
        self.tickets: dict[str, list[dict[str, Any]]] = seed["tickets"]
        with self._log_lock:
            self.access_log: list[dict[str, Any]] = []
        with self.db.transaction(write=True) as tx:
            for table in ("idempotency_records", "ledger_entries", "refunds", "escalations",
                          "orders", "counters"):
                tx.execute(f"DELETE FROM {table}")
            for order in self.orders.values():
                tx.execute("INSERT INTO orders (order_id, customer_id) VALUES (:order_id, :customer_id)",
                           {"order_id": order["orderId"], "customer_id": order["customerId"]})
            for name, value in _INITIAL_COUNTERS.items():
                tx.execute("INSERT INTO counters (name, value) VALUES (:name, :value)",
                           {"name": name, "value": value})

    # ---- clock and counters ---------------------------------------------------------------

    def _clock(self, tx: Transaction) -> datetime:
        row = tx.fetchone("SELECT value FROM counters WHERE name = 'clock'")
        return AS_OF + timedelta(seconds=int(row["value"]))

    def _next_value(self, tx: Transaction, name: str) -> int:
        row = tx.fetchone(
            "UPDATE counters SET value = value + 1 WHERE name = :name RETURNING value",
            {"name": name},
        )
        return int(row["value"])

    def _tick(self, tx: Transaction) -> datetime:
        """Advance the backend clock by one second for a mutation and return the new time."""
        return AS_OF + timedelta(seconds=self._next_value(tx, "clock"))

    def now(self) -> datetime:
        """Backend clock: AS_OF plus one second per committed mutation."""
        with self.db.transaction() as tx:
            return self._clock(tx)

    def advance_clock(self, seconds: int) -> None:
        """Test hook: move the backend clock forward (for example past the idempotency TTL)."""
        with self.db.transaction(write=True) as tx:
            tx.execute("UPDATE counters SET value = value + :seconds WHERE name = 'clock'",
                       {"seconds": seconds})

    def _checkpoint(self, stage: str) -> None:
        """Test seam inside write transactions; tests patch it to pause or crash mid-write."""

    # ---- access control -------------------------------------------------------------------

    def _log(self, action: str, resource: str, on_behalf_of: str | None, decision: str) -> None:
        with self._log_lock:
            self.access_log.append(
                {
                    "seq": len(self.access_log) + 1,
                    "action": action,
                    "resource": resource,
                    "onBehalfOf": on_behalf_of,
                    "decision": decision,
                }
            )

    def _customer(self, customer_id: str, on_behalf_of: str | None, action: str) -> dict[str, Any]:
        customer = self.customers.get(customer_id)
        if customer is None or (on_behalf_of is not None and customer_id != on_behalf_of):
            decision = "not_found" if customer is None else "denied_not_owner"
            self._log(action, customer_id, on_behalf_of, decision)
            raise StoreError(404, "CUSTOMER_NOT_FOUND", f"Customer {customer_id} was not found.")
        self._log(action, customer_id, on_behalf_of, "allowed")
        return customer

    def _order(self, order_id: str, on_behalf_of: str | None, action: str) -> dict[str, Any]:
        order = self.orders.get(order_id)
        if order is None or (on_behalf_of is not None and order["customerId"] != on_behalf_of):
            decision = "not_found" if order is None else "denied_not_owner"
            self._log(action, order_id, on_behalf_of, decision)
            # Identical response for "missing" and "someone else's": no existence oracle.
            raise StoreError(404, "ORDER_NOT_FOUND", f"Order {order_id} was not found.")
        self._log(action, order_id, on_behalf_of, "allowed")
        return order

    def _orders_of(self, customer_id: str) -> list[dict[str, Any]]:
        owned = [o for o in self.orders.values() if o["customerId"] == customer_id]
        return sorted(owned, key=lambda o: o["createdAt"], reverse=True)

    # ---- database-backed views -------------------------------------------------------------

    def _refunds(self, tx: Transaction, order_id: str) -> list[dict[str, Any]]:
        rows = tx.fetchall(
            "SELECT * FROM refunds WHERE order_id = :order_id ORDER BY created_at, refund_id",
            {"order_id": order_id},
        )
        return [_refund_from_row(row) for row in rows]

    def _transactions(self, tx: Transaction, order_id: str) -> list[dict[str, Any]]:
        """Seed payment transactions plus a REFUND transaction for every refund row."""
        transactions = copy.deepcopy(self._seed_transactions[order_id])
        by_id = {t["transactionId"]: t for t in transactions}
        rows = tx.fetchall(
            "SELECT refund_id, amount_cents, created_at, capture_transaction_id FROM refunds "
            "WHERE order_id = :order_id ORDER BY created_at, refund_id",
            {"order_id": order_id},
        )
        for row in rows:
            capture = by_id[row["capture_transaction_id"]]
            transactions.append(
                {
                    "transactionId": f"txn_rf{row['refund_id'][3:]}",
                    "orderId": order_id,
                    "type": "REFUND",
                    "status": "PENDING",
                    "amountCents": int(row["amount_cents"]),
                    "currency": "USD",
                    "createdAt": row["created_at"],
                    "paymentMethod": dict(capture["paymentMethod"]),
                    "descriptor": capture["descriptor"],
                    "relatedTransactionId": capture["transactionId"],
                    "refundId": row["refund_id"],
                }
            )
        return transactions

    @property
    def refunds(self) -> Mapping[str, list[dict[str, Any]]]:
        return _PerOrderView(self, self._refunds)

    @property
    def transactions(self) -> Mapping[str, list[dict[str, Any]]]:
        return _PerOrderView(self, self._transactions)

    @property
    def ledger(self) -> list[dict[str, Any]]:
        with self.db.transaction() as tx:
            rows = tx.fetchall("SELECT payload FROM ledger_entries ORDER BY recorded_at, entry_id")
        return [{"seq": seq, **json.loads(row["payload"])} for seq, row in enumerate(rows, start=1)]

    @property
    def escalations(self) -> list[dict[str, Any]]:
        with self.db.transaction() as tx:
            rows = tx.fetchall("SELECT * FROM escalations ORDER BY created_at, escalation_id")
        return [_escalation_from_row(row) for row in rows]

    # ---- reads ------------------------------------------------------------------------------

    def get_customer(self, customer_id: str, on_behalf_of: str | None) -> dict[str, Any]:
        return copy.deepcopy(self._customer(customer_id, on_behalf_of, "customer.read"))

    def list_orders(self, customer_id: str, on_behalf_of: str | None) -> list[dict[str, Any]]:
        """Full order records (legacy contract: GET /api/orders/user/{id})."""
        if on_behalf_of is None:
            self._log("orders.list", customer_id, None, "allowed")
        else:
            self._customer(customer_id, on_behalf_of, "orders.list")
        return copy.deepcopy(self._orders_of(customer_id))

    def list_order_summaries(self, customer_id: str, on_behalf_of: str | None) -> list[dict[str, Any]]:
        self._customer(customer_id, on_behalf_of, "orders.list")
        return [
            {
                "orderId": order["orderId"],
                "status": order["status"],
                "createdAt": order["createdAt"],
                "totalCents": order["totalCents"],
                "itemCount": sum(item["quantity"] for item in order["items"]),
                "items": [item["productName"] for item in order["items"]],
            }
            for order in self._orders_of(customer_id)
        ]

    def list_tickets(self, customer_id: str, on_behalf_of: str | None) -> list[dict[str, Any]]:
        self._customer(customer_id, on_behalf_of, "tickets.list")
        return copy.deepcopy(self.tickets.get(customer_id, []))

    def get_order(self, order_id: str, on_behalf_of: str | None) -> dict[str, Any]:
        return copy.deepcopy(self._order(order_id, on_behalf_of, "order.read"))

    def get_payments(self, order_id: str, on_behalf_of: str | None) -> dict[str, Any]:
        self._order(order_id, on_behalf_of, "payments.read")
        with self.db.transaction() as tx:
            return {"orderId": order_id, "asOf": iso(self._clock(tx)),
                    "transactions": self._transactions(tx, order_id)}

    def get_shipments(self, order_id: str, on_behalf_of: str | None) -> dict[str, Any]:
        self._order(order_id, on_behalf_of, "shipments.read")
        return {"orderId": order_id, "asOf": iso(self.now()),
                "shipments": copy.deepcopy(self.shipments[order_id])}

    def get_refunds(self, order_id: str, on_behalf_of: str | None) -> dict[str, Any]:
        self._order(order_id, on_behalf_of, "refunds.read")
        with self.db.transaction() as tx:
            return {"orderId": order_id, "refunds": self._refunds(tx, order_id)}

    # ---- refund rules -----------------------------------------------------------------------

    def _money(self, tx: Transaction, order_id: str) -> dict[str, int]:
        seed = self._seed_transactions[order_id]
        captured = sum(t["amountCents"] for t in seed
                       if t["type"] == "CAPTURE" and t["status"] == "SUCCEEDED")
        disputed = sum(t["amountCents"] for t in seed
                       if t["type"] == "CHARGEBACK" and t["status"] in ("OPEN", "LOST"))
        row = tx.fetchone("SELECT COALESCE(SUM(amount_cents), 0) AS total FROM refunds "
                          "WHERE order_id = :order_id", {"order_id": order_id})
        refunded = int(row["total"])
        return {
            "capturedCents": captured,
            "refundedCents": refunded,
            "disputedCents": disputed,
            "refundableCents": max(0, captured - refunded - disputed),
        }

    def _refunded_for(self, tx: Transaction, order_id: str, reason_code: str) -> int:
        row = tx.fetchone(
            "SELECT COALESCE(SUM(amount_cents), 0) AS total FROM refunds "
            "WHERE order_id = :order_id AND reason_code = :reason_code",
            {"order_id": order_id, "reason_code": reason_code},
        )
        return int(row["total"])

    def _lost_shipments(self, order_id: str, now: datetime) -> list[dict[str, Any]]:
        """Undelivered, past the promised date, and no carrier scan for 5+ days."""
        lost = []
        for shipment in self.shipments[order_id]:
            if (
                shipment["direction"] != "OUTBOUND"
                or shipment["status"] == "DELIVERED"
                or not shipment["promisedBy"]
            ):
                continue
            stale = now - parse_iso(shipment["lastScanAt"]) >= timedelta(
                days=LOST_PACKAGE_MIN_DAYS_WITHOUT_SCAN
            )
            if stale and now > parse_iso(shipment["promisedBy"]):
                lost.append(shipment)
        return lost

    def _reason_limit(
        self,
        tx: Transaction,
        order: dict[str, Any],
        reason_code: str,
        line_ids: list[str],
        blockers: list[dict[str, str]],
        now: datetime,
    ) -> tuple[int, str]:
        """Maximum refundable for this reason, plus a one-line explanation of the basis."""
        order_id = order["orderId"]

        if reason_code == "duplicate_charge":
            captures = [
                t for t in self._seed_transactions[order_id]
                if t["type"] == "CAPTURE" and t["status"] == "SUCCEEDED"
            ]
            captured = sum(t["amountCents"] for t in captures)
            limit = captured - order["totalCents"] - self._refunded_for(tx, order_id, reason_code)
            basis = (
                f"{len(captures)} successful capture(s) totalling {_usd(captured)} against an "
                f"order total of {_usd(order['totalCents'])}."
            )
            if limit <= 0:
                blockers.append(_blocker(
                    "NO_DUPLICATE_CAPTURE",
                    "Captured funds do not exceed the order total (net of earlier duplicate "
                    "refunds), so nothing is owed. Voided or pending authorizations are holds, "
                    "not charges.",
                ))
            return max(limit, 0), basis

        if reason_code == "lost_package":
            lost = self._lost_shipments(order_id, now)
            if not lost:
                blockers.append(_blocker(
                    "NOT_LOST",
                    "No outbound shipment meets the lost-package rule: undelivered, past its "
                    f"promised date, and no carrier scan for {LOST_PACKAGE_MIN_DAYS_WITHOUT_SCAN}+ "
                    "days.",
                ))
                return 0, "No lost shipments."
            lost_lines = {line for shipment in lost for line in shipment["lineIds"]}
            value = sum(i["lineTotalCents"] for i in order["items"] if i["lineId"] in lost_lines)
            if lost_lines == {item["lineId"] for item in order["items"]}:
                value += order["shippingCents"]
            limit = value - self._refunded_for(tx, order_id, reason_code)
            basis = (
                f"Lost shipment(s) {', '.join(s['shipmentId'] for s in lost)} covering "
                f"{len(lost_lines)} line(s) worth {_usd(value)} including tax."
            )
            return max(limit, 0), basis

        if reason_code == "damaged_item_keep":
            if not line_ids:
                blockers.append(_blocker(
                    "LINE_IDS_REQUIRED", "damaged_item_keep needs the lineIds of the damaged items."
                ))
                return 0, ""
            by_id = {item["lineId"]: item for item in order["items"]}
            unknown = [line for line in line_ids if line not in by_id]
            if unknown:
                blockers.append(_blocker(
                    "UNKNOWN_LINE", f"Line(s) not on this order: {', '.join(unknown)}."
                ))
                return 0, ""
            delivered = {
                line
                for shipment in self.shipments[order_id]
                if shipment["direction"] == "OUTBOUND" and shipment["status"] == "DELIVERED"
                for line in shipment["lineIds"]
            }
            undelivered = [line for line in line_ids if line not in delivered]
            if undelivered:
                blockers.append(_blocker(
                    "LINE_NOT_DELIVERED",
                    f"Line(s) not delivered yet, so they cannot be damaged-on-arrival: "
                    f"{', '.join(undelivered)}.",
                ))
                return 0, ""
            value = sum(by_id[line]["lineTotalCents"] for line in set(line_ids))
            cap = int((Decimal(value) * DAMAGED_KEEP_MAX_FRACTION).to_integral_value(ROUND_FLOOR))
            limit = cap - self._refunded_for(tx, order_id, reason_code)
            basis = (
                f"Up to 25% of {_usd(value)} (item price plus tax) for damaged item(s) the "
                "customer keeps."
            )
            return max(limit, 0), basis

        blockers.append(_blocker(
            "UNKNOWN_REASON", f"reasonCode must be one of: {', '.join(REFUND_REASONS)}."
        ))
        return 0, ""

    def _evaluate_refund(
        self,
        tx: Transaction,
        order: dict[str, Any],
        amount_cents: int,
        reason_code: str,
        line_ids: list[str],
        now: datetime,
    ) -> dict[str, Any]:
        blockers: list[dict[str, str]] = []
        limit, basis = self._reason_limit(tx, order, reason_code, line_ids, blockers, now)
        reason_ok = not blockers
        money = self._money(tx, order["orderId"])

        if amount_cents <= 0:
            blockers.append(_blocker("INVALID_AMOUNT", "amountCents must be a positive integer."))
        for dispute in self._seed_transactions[order["orderId"]]:
            if dispute["type"] == "CHARGEBACK" and dispute["status"] == "OPEN":
                blockers.append(_blocker(
                    "DISPUTE_OPEN",
                    f"Chargeback {dispute['disputeId']} is open (respond by "
                    f"{dispute['respondBy'][:10]}). Refunds are frozen while a dispute is open; "
                    "escalate to the disputes queue instead.",
                ))
        if reason_ok and amount_cents > limit:
            blockers.append(_blocker(
                "EXCEEDS_REASON_MAX",
                f"Requested {_usd(amount_cents)} exceeds the {_usd(limit)} allowed for "
                f"{reason_code}.",
            ))
        if amount_cents > money["refundableCents"]:
            blockers.append(_blocker(
                "EXCEEDS_REFUNDABLE",
                f"Requested {_usd(amount_cents)} exceeds the {_usd(money['refundableCents'])} "
                "still refundable on this order.",
            ))

        return {
            "orderId": order["orderId"],
            "reasonCode": reason_code,
            "amountCents": amount_cents,
            "eligible": not blockers,
            "maxForReasonCents": limit,
            "basis": basis,
            **money,
            "blockingReasons": blockers,
            "asOf": iso(now),
        }

    def preview_refund(
        self,
        on_behalf_of: str | None,
        *,
        order_id: str,
        amount_cents: int,
        reason_code: str,
        line_ids: list[str],
    ) -> dict[str, Any]:
        """Read-only check of a proposed refund against every rule create_refund enforces."""
        order = self._order(order_id, on_behalf_of, "refund.preview")
        with self.db.transaction() as tx:
            return self._evaluate_refund(tx, order, amount_cents, reason_code, line_ids,
                                         self._clock(tx))

    # ---- idempotent writes ------------------------------------------------------------------

    def _find_replay(
        self, tx: Transaction, scope: tuple[str, str, str], request_hash: str, now: datetime
    ) -> Outcome | None:
        customer_id, endpoint, key = scope
        params = {"customer_id": customer_id, "endpoint": endpoint, "key": key}
        row = tx.fetchone(
            "SELECT request_hash, status_code, response_body, expires_at FROM idempotency_records "
            "WHERE customer_id = :customer_id AND endpoint = :endpoint AND idempotency_key = :key",
            params,
        )
        if row is None:
            return None
        if row["expires_at"] <= iso(now):
            tx.execute(
                "DELETE FROM idempotency_records WHERE customer_id = :customer_id "
                "AND endpoint = :endpoint AND idempotency_key = :key",
                params,
            )
            return None
        if row["request_hash"] != request_hash:
            raise StoreError(
                409, "IDEMPOTENCY_KEY_REUSED",
                "This Idempotency-Key was already used with a different request.",
            )
        return Outcome(int(row["status_code"]), json.loads(row["response_body"]), replayed=True)

    def _remember(
        self, tx: Transaction, scope: tuple[str, str, str], request_hash: str, outcome: Outcome
    ) -> None:
        customer_id, endpoint, key = scope
        now = self._clock(tx)
        tx.execute(
            "INSERT INTO idempotency_records (customer_id, endpoint, idempotency_key, request_hash, "
            "status_code, response_body, created_at, expires_at) VALUES (:customer_id, :endpoint, "
            ":key, :request_hash, :status_code, :response_body, :created_at, :expires_at)",
            {
                "customer_id": customer_id,
                "endpoint": endpoint,
                "key": key,
                "request_hash": request_hash,
                "status_code": outcome.status_code,
                "response_body": json.dumps(outcome.body, ensure_ascii=False),
                "created_at": iso(now),
                "expires_at": iso(now + IDEMPOTENCY_TTL),
            },
        )

    def _idempotent_write(
        self,
        scope: tuple[str, str, str],
        request_hash: str,
        *,
        lock_order_id: str | None,
        work: Callable[[Transaction], Outcome],
    ) -> Outcome:
        """Replay a stored outcome, or run `work` and store its outcome, in one transaction.

        A concurrent request with the same key waits on the lock and then replays the first
        request's stored outcome. On Postgres without an order lock, the loser of that race
        hits the unique constraint instead; running once more turns that into a replay.
        """
        for attempt in range(2):
            try:
                with self.db.transaction(write=True) as tx:
                    if lock_order_id is not None:
                        tx.lock_order(lock_order_id)
                    replay = self._find_replay(tx, scope, request_hash, self._clock(tx))
                    if replay is not None:
                        return replay
                    outcome = work(tx)
                    self._remember(tx, scope, request_hash, outcome)
                    return outcome
            except UniqueViolation:
                if attempt == 1:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    def _record_ledger(
        self, tx: Transaction, entry_id: str, now: datetime, customer_id: str,
        order_id: str | None, payload: dict[str, Any],
    ) -> None:
        tx.execute(
            "INSERT INTO ledger_entries (entry_id, entry_type, recorded_at, customer_id, order_id, "
            "payload) VALUES (:entry_id, :entry_type, :recorded_at, :customer_id, :order_id, "
            ":payload)",
            {
                "entry_id": entry_id,
                "entry_type": payload["type"],
                "recorded_at": iso(now),
                "customer_id": customer_id,
                "order_id": order_id,
                "payload": json.dumps(payload, ensure_ascii=False),
            },
        )

    # ---- mutations --------------------------------------------------------------------------

    def create_refund(
        self,
        on_behalf_of: str | None,
        *,
        idempotency_key: str,
        order_id: str,
        amount_cents: int,
        reason_code: str,
        line_ids: list[str],
        case_id: str,
        approved_by: str,
        note: str = "",
    ) -> Outcome:
        """Create a refund, or replay the stored outcome for this key."""
        _check_idempotency_key(idempotency_key)
        order = self._order(order_id, on_behalf_of, "refund.create")
        if not approved_by.strip():
            raise StoreError(
                422, "APPROVER_REQUIRED",
                "Agent-initiated refunds must name the human who approved them.",
            )
        request_hash = _hash_request({
            "orderId": order_id, "amountCents": amount_cents, "reasonCode": reason_code,
            "lineIds": list(line_ids), "caseId": case_id, "approvedBy": approved_by, "note": note,
        })
        customer_id = order["customerId"]

        def work(tx: Transaction) -> Outcome:
            evaluation = self._evaluate_refund(tx, order, amount_cents, reason_code, line_ids,
                                               self._clock(tx))
            self._checkpoint("balance_checked")
            if not evaluation["eligible"]:
                return Outcome(409, {
                    "errorCode": "REFUND_BLOCKED",
                    "message": "Refund rejected by policy checks.",
                    "blockingReasons": evaluation["blockingReasons"],
                })

            now = self._tick(tx)
            refund_id = f"RF-{self._next_value(tx, 'refund')}"
            capture = next(
                t for t in reversed(self._seed_transactions[order_id])
                if t["type"] == "CAPTURE" and t["status"] == "SUCCEEDED"
            )
            refund = {
                "refundId": refund_id,
                "orderId": order_id,
                "customerId": customer_id,
                "amountCents": amount_cents,
                "currency": "USD",
                "reasonCode": reason_code,
                "lineIds": list(line_ids),
                "status": "PENDING",
                "createdAt": iso(now),
                "expectedSettlementBy": iso(now + timedelta(days=REFUND_SETTLEMENT_DAYS)),
                "approvedBy": approved_by,
                "caseId": case_id,
                "note": note,
            }
            tx.execute(
                "INSERT INTO refunds (refund_id, order_id, customer_id, amount_cents, currency, "
                "reason_code, line_ids, status, created_at, expected_settlement_by, approved_by, "
                "case_id, note, capture_transaction_id) VALUES (:refund_id, :order_id, "
                ":customer_id, :amount_cents, :currency, :reason_code, :line_ids, :status, "
                ":created_at, :expected_settlement_by, :approved_by, :case_id, :note, "
                ":capture_transaction_id)",
                {
                    "refund_id": refund_id, "order_id": order_id, "customer_id": customer_id,
                    "amount_cents": amount_cents, "currency": "USD", "reason_code": reason_code,
                    "line_ids": json.dumps(list(line_ids)), "status": "PENDING",
                    "created_at": refund["createdAt"],
                    "expected_settlement_by": refund["expectedSettlementBy"],
                    "approved_by": approved_by, "case_id": case_id, "note": note,
                    "capture_transaction_id": capture["transactionId"],
                },
            )
            self._checkpoint("refund_inserted")
            self._record_ledger(tx, f"refund.created:{refund_id}", now, customer_id, order_id, {
                "type": "refund.created",
                "at": iso(now),
                "refundId": refund_id,
                "orderId": order_id,
                "customerId": customer_id,
                "amountCents": amount_cents,
                "reasonCode": reason_code,
                "approvedBy": approved_by,
                "caseId": case_id,
                "idempotencyKey": idempotency_key,
            })
            return Outcome(201, refund)

        return self._idempotent_write((customer_id, REFUNDS_ENDPOINT, idempotency_key),
                                      request_hash, lock_order_id=order_id, work=work)

    def create_escalation(
        self,
        on_behalf_of: str,
        *,
        idempotency_key: str,
        queue: str,
        priority: str,
        summary: str,
        case_id: str,
        order_id: str | None = None,
    ) -> Outcome:
        """Open a ticket in a human team's queue, or replay the stored outcome for this key."""
        _check_idempotency_key(idempotency_key)
        self._customer(on_behalf_of, on_behalf_of, "escalation.create")
        if order_id is not None:
            self._order(order_id, on_behalf_of, "escalation.create")
        if queue not in ESCALATION_SLA_HOURS:
            raise StoreError(
                422, "UNKNOWN_QUEUE", f"queue must be one of: {', '.join(ESCALATION_SLA_HOURS)}."
            )
        request_hash = _hash_request({
            "queue": queue, "priority": priority, "summary": summary, "caseId": case_id,
            "orderId": order_id,
        })

        def work(tx: Transaction) -> Outcome:
            now = self._tick(tx)
            escalation_id = f"ESC-{self._next_value(tx, 'escalation')}"
            escalation = {
                "escalationId": escalation_id,
                "customerId": on_behalf_of,
                "orderId": order_id,
                "queue": queue,
                "priority": priority,
                "status": "OPEN",
                "summary": summary,
                "caseId": case_id,
                "createdAt": iso(now),
                "firstResponseDueBy": iso(now + timedelta(hours=ESCALATION_SLA_HOURS[queue])),
            }
            tx.execute(
                "INSERT INTO escalations (escalation_id, customer_id, order_id, queue, priority, "
                "status, summary, case_id, created_at, first_response_due_by) VALUES "
                "(:escalation_id, :customer_id, :order_id, :queue, :priority, :status, :summary, "
                ":case_id, :created_at, :first_response_due_by)",
                {
                    "escalation_id": escalation_id, "customer_id": on_behalf_of,
                    "order_id": order_id, "queue": queue, "priority": priority, "status": "OPEN",
                    "summary": summary, "case_id": case_id, "created_at": escalation["createdAt"],
                    "first_response_due_by": escalation["firstResponseDueBy"],
                },
            )
            self._checkpoint("escalation_inserted")
            self._record_ledger(tx, f"escalation.created:{escalation_id}", now, on_behalf_of,
                                order_id, {
                                    "type": "escalation.created",
                                    "at": iso(now),
                                    "escalationId": escalation_id,
                                    "orderId": order_id,
                                    "customerId": on_behalf_of,
                                    "queue": queue,
                                    "priority": priority,
                                    "caseId": case_id,
                                    "idempotencyKey": idempotency_key,
                                })
            return Outcome(201, escalation)

        return self._idempotent_write((on_behalf_of, ESCALATIONS_ENDPOINT, idempotency_key),
                                      request_hash, lock_order_id=order_id, work=work)
