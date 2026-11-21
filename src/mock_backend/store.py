"""In-memory CloudOrder state plus the money rules the real services enforce.

The agent never imports this module; it only talks to the HTTP API in app.py. Rules live
here, not in prompts, so a confused or manipulated agent still cannot move money that
policy does not allow.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta
from decimal import ROUND_FLOOR, Decimal
from typing import Any

from src.mock_backend.data import AS_OF, build_seed, iso, parse_iso

REFUND_REASONS = ("duplicate_charge", "lost_package", "damaged_item_keep")
ESCALATION_SLA_HOURS = {"disputes": 48, "tier2_support": 24, "trust_safety": 72}
LOST_PACKAGE_MIN_DAYS_WITHOUT_SCAN = 5
DAMAGED_KEEP_MAX_FRACTION = Decimal("0.25")
REFUND_SETTLEMENT_DAYS = 7


class StoreError(Exception):
    """A business or access error that maps onto an HTTP status and error code."""

    def __init__(self, status: int, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details


def _blocker(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _usd(cents: int) -> str:
    return f"${cents / 100:,.2f}"


class CloudOrderStore:
    """Single-process store.

    Methods are synchronous and are called from async endpoints, so each request's
    read-check-write runs without interleaving. A real service would use database
    transactions for the same guarantee.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        seed = build_seed()
        self.customers: dict[str, dict[str, Any]] = seed["customers"]
        self.orders: dict[str, dict[str, Any]] = seed["orders"]
        self.transactions: dict[str, list[dict[str, Any]]] = seed["transactions"]
        self.shipments: dict[str, list[dict[str, Any]]] = seed["shipments"]
        self.refunds: dict[str, list[dict[str, Any]]] = seed["refunds"]
        self.tickets: dict[str, list[dict[str, Any]]] = seed["tickets"]
        self.escalations: list[dict[str, Any]] = []
        self.ledger: list[dict[str, Any]] = []
        self.access_log: list[dict[str, Any]] = []
        self._idempotency: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        self._ticks = 0
        self._next_refund = 90001
        self._next_escalation = 50001

    def now(self) -> datetime:
        """Backend clock: frozen at AS_OF, advanced one second per mutation."""
        return AS_OF + timedelta(seconds=self._ticks)

    # ---- access control ---------------------------------------------------------

    def _log(self, action: str, resource: str, on_behalf_of: str | None, decision: str) -> None:
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

    # ---- reads ------------------------------------------------------------------

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
        return {
            "orderId": order_id,
            "asOf": iso(self.now()),
            "transactions": copy.deepcopy(self.transactions[order_id]),
        }

    def get_shipments(self, order_id: str, on_behalf_of: str | None) -> dict[str, Any]:
        self._order(order_id, on_behalf_of, "shipments.read")
        return {
            "orderId": order_id,
            "asOf": iso(self.now()),
            "shipments": copy.deepcopy(self.shipments[order_id]),
        }

    def get_refunds(self, order_id: str, on_behalf_of: str | None) -> dict[str, Any]:
        self._order(order_id, on_behalf_of, "refunds.read")
        return {"orderId": order_id, "refunds": copy.deepcopy(self.refunds[order_id])}

    # ---- refund rules -----------------------------------------------------------

    def _money(self, order_id: str) -> dict[str, int]:
        transactions = self.transactions[order_id]
        captured = sum(
            t["amountCents"] for t in transactions
            if t["type"] == "CAPTURE" and t["status"] == "SUCCEEDED"
        )
        refunded = sum(r["amountCents"] for r in self.refunds[order_id])
        disputed = sum(
            t["amountCents"] for t in transactions
            if t["type"] == "CHARGEBACK" and t["status"] in ("OPEN", "LOST")
        )
        return {
            "capturedCents": captured,
            "refundedCents": refunded,
            "disputedCents": disputed,
            "refundableCents": max(0, captured - refunded - disputed),
        }

    def _refunded_for(self, order_id: str, reason_code: str) -> int:
        return sum(r["amountCents"] for r in self.refunds[order_id] if r["reasonCode"] == reason_code)

    def _lost_shipments(self, order_id: str) -> list[dict[str, Any]]:
        """Undelivered, past the promised date, and no carrier scan for 5+ days."""
        now = self.now()
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
        order: dict[str, Any],
        reason_code: str,
        line_ids: list[str],
        blockers: list[dict[str, str]],
    ) -> tuple[int, str]:
        """Maximum refundable for this reason, plus a one-line explanation of the basis."""
        order_id = order["orderId"]

        if reason_code == "duplicate_charge":
            captures = [
                t for t in self.transactions[order_id]
                if t["type"] == "CAPTURE" and t["status"] == "SUCCEEDED"
            ]
            captured = sum(t["amountCents"] for t in captures)
            limit = captured - order["totalCents"] - self._refunded_for(order_id, reason_code)
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
            lost = self._lost_shipments(order_id)
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
            limit = value - self._refunded_for(order_id, reason_code)
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
            limit = cap - self._refunded_for(order_id, reason_code)
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
        self, order: dict[str, Any], amount_cents: int, reason_code: str, line_ids: list[str]
    ) -> dict[str, Any]:
        blockers: list[dict[str, str]] = []
        limit, basis = self._reason_limit(order, reason_code, line_ids, blockers)
        reason_ok = not blockers
        money = self._money(order["orderId"])

        if amount_cents <= 0:
            blockers.append(_blocker("INVALID_AMOUNT", "amountCents must be a positive integer."))
        for dispute in self.transactions[order["orderId"]]:
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
            "asOf": iso(self.now()),
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
        return self._evaluate_refund(order, amount_cents, reason_code, line_ids)

    # ---- mutations --------------------------------------------------------------

    def _replay(self, kind: str, key: str, request: dict[str, Any]) -> dict[str, Any] | None:
        stored = self._idempotency.get((kind, key))
        if stored is None:
            return None
        fingerprint, response = stored
        if fingerprint != json.dumps(request, sort_keys=True):
            raise StoreError(
                409, "IDEMPOTENCY_KEY_REUSED",
                "This Idempotency-Key was already used with a different request.",
            )
        return copy.deepcopy(response)

    def _remember(self, kind: str, key: str, request: dict[str, Any], response: dict[str, Any]) -> None:
        self._idempotency[(kind, key)] = (json.dumps(request, sort_keys=True), copy.deepcopy(response))

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
    ) -> tuple[dict[str, Any], bool]:
        """Create a refund. Returns (refund, replayed). Re-checks every preview rule."""
        request = {
            "onBehalfOf": on_behalf_of,
            "orderId": order_id,
            "amountCents": amount_cents,
            "reasonCode": reason_code,
            "lineIds": sorted(line_ids),
            "caseId": case_id,
            "approvedBy": approved_by,
        }
        replay = self._replay("refund", idempotency_key, request)
        if replay is not None:
            return replay, True

        order = self._order(order_id, on_behalf_of, "refund.create")
        if not approved_by.strip():
            raise StoreError(
                422, "APPROVER_REQUIRED",
                "Agent-initiated refunds must name the human who approved them.",
            )
        evaluation = self._evaluate_refund(order, amount_cents, reason_code, line_ids)
        if not evaluation["eligible"]:
            raise StoreError(
                409, "REFUND_BLOCKED", "Refund rejected by policy checks.",
                blockingReasons=evaluation["blockingReasons"],
            )

        self._ticks += 1
        now = self.now()
        refund_id = f"RF-{self._next_refund}"
        self._next_refund += 1
        refund = {
            "refundId": refund_id,
            "orderId": order_id,
            "customerId": order["customerId"],
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
        self.refunds[order_id].append(refund)

        capture = next(
            t for t in reversed(self.transactions[order_id])
            if t["type"] == "CAPTURE" and t["status"] == "SUCCEEDED"
        )
        self.transactions[order_id].append(
            {
                "transactionId": f"txn_rf{refund_id[3:]}",
                "orderId": order_id,
                "type": "REFUND",
                "status": "PENDING",
                "amountCents": amount_cents,
                "currency": "USD",
                "createdAt": iso(now),
                "paymentMethod": dict(capture["paymentMethod"]),
                "descriptor": capture["descriptor"],
                "relatedTransactionId": capture["transactionId"],
                "refundId": refund_id,
            }
        )
        self.ledger.append(
            {
                "seq": len(self.ledger) + 1,
                "type": "refund.created",
                "at": iso(now),
                "refundId": refund_id,
                "orderId": order_id,
                "customerId": order["customerId"],
                "amountCents": amount_cents,
                "reasonCode": reason_code,
                "approvedBy": approved_by,
                "caseId": case_id,
                "idempotencyKey": idempotency_key,
            }
        )
        self._remember("refund", idempotency_key, request, refund)
        return copy.deepcopy(refund), False

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
    ) -> tuple[dict[str, Any], bool]:
        """Open a ticket in a human team's queue. Returns (escalation, replayed)."""
        request = {
            "onBehalfOf": on_behalf_of,
            "queue": queue,
            "priority": priority,
            "summary": summary,
            "caseId": case_id,
            "orderId": order_id,
        }
        replay = self._replay("escalation", idempotency_key, request)
        if replay is not None:
            return replay, True

        self._customer(on_behalf_of, on_behalf_of, "escalation.create")
        if order_id is not None:
            self._order(order_id, on_behalf_of, "escalation.create")
        if queue not in ESCALATION_SLA_HOURS:
            raise StoreError(
                422, "UNKNOWN_QUEUE", f"queue must be one of: {', '.join(ESCALATION_SLA_HOURS)}."
            )

        self._ticks += 1
        now = self.now()
        escalation_id = f"ESC-{self._next_escalation}"
        self._next_escalation += 1
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
        self.escalations.append(escalation)
        self.ledger.append(
            {
                "seq": len(self.ledger) + 1,
                "type": "escalation.created",
                "at": iso(now),
                "escalationId": escalation_id,
                "orderId": order_id,
                "customerId": on_behalf_of,
                "queue": queue,
                "priority": priority,
                "caseId": case_id,
                "idempotencyKey": idempotency_key,
            }
        )
        self._remember("escalation", idempotency_key, request, escalation)
        return copy.deepcopy(escalation), False
