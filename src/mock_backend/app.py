"""HTTP API for the mock CloudOrder backend.

Run standalone:  uvicorn src.mock_backend.app:app --port 8200
The deep-agent path can also call it in-process through httpx.ASGITransport.

Contract conventions: camelCase JSON, integer cents, `X-On-Behalf-Of` on every
customer-delegated call, and `Idempotency-Key` on every mutation. The legacy order
endpoints keep the shape the existing order/ReAct tools already read.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from src.mock_backend.data import iso
from src.mock_backend.store import CloudOrderStore, StoreError


class _CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, extra="forbid")


class RefundPreviewRequest(_CamelModel):
    order_id: str
    amount_cents: int
    reason_code: str
    line_ids: list[str] = Field(default_factory=list)


class RefundCreateRequest(RefundPreviewRequest):
    case_id: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    note: str = Field(default="", max_length=1000)


class EscalationCreateRequest(_CamelModel):
    queue: Literal["disputes", "tier2_support", "trust_safety"]
    priority: Literal["low", "normal", "high", "urgent"] = "normal"
    summary: str = Field(min_length=10, max_length=2000)
    case_id: str = Field(min_length=1)
    order_id: str | None = None


OnBehalfOf = Annotated[str | None, Header(alias="X-On-Behalf-Of")]
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key")]


def _require_on_behalf_of(value: str | None) -> str:
    if not value:
        raise StoreError(
            401, "ON_BEHALF_OF_REQUIRED",
            "Customer-scoped endpoints require the X-On-Behalf-Of header.",
        )
    return value


def _require_idempotency_key(value: str | None) -> str:
    if not value:
        raise StoreError(
            400, "IDEMPOTENCY_KEY_REQUIRED", "Mutations require an Idempotency-Key header."
        )
    return value


def _created(resource: dict[str, Any], replayed: bool) -> JSONResponse:
    if replayed:
        return JSONResponse(resource, status_code=200, headers={"Idempotent-Replayed": "true"})
    return JSONResponse(resource, status_code=201)


def create_app(store: CloudOrderStore | None = None) -> FastAPI:
    backend = store or CloudOrderStore()
    app = FastAPI(
        title="CloudOrder mock backend",
        version="1.0.0",
        description="Stand-in for CloudOrder's private order, payment, fulfillment, refund, "
                    "and support services.",
    )
    app.state.store = backend

    @app.exception_handler(StoreError)
    async def _store_error(_: Request, exc: StoreError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content={"errorCode": exc.code, "message": exc.message, **exc.details},
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "asOf": iso(backend.now())}

    # Legacy order-service contract (X-On-Behalf-Of optional, enforced when present).
    @app.get("/api/orders/user/{customer_id}")
    async def legacy_user_orders(customer_id: str, on_behalf_of: OnBehalfOf = None) -> list[dict[str, Any]]:
        return backend.list_orders(customer_id, on_behalf_of)

    @app.get("/api/orders/{order_id}")
    async def get_order(order_id: str, on_behalf_of: OnBehalfOf = None) -> dict[str, Any]:
        return backend.get_order(order_id, on_behalf_of)

    # Customer-delegated endpoints (X-On-Behalf-Of required).
    @app.get("/api/customers/{customer_id}")
    async def get_customer(customer_id: str, on_behalf_of: OnBehalfOf = None) -> dict[str, Any]:
        return backend.get_customer(customer_id, _require_on_behalf_of(on_behalf_of))

    @app.get("/api/customers/{customer_id}/orders")
    async def list_orders(customer_id: str, on_behalf_of: OnBehalfOf = None) -> dict[str, Any]:
        orders = backend.list_order_summaries(customer_id, _require_on_behalf_of(on_behalf_of))
        return {"customerId": customer_id, "orders": orders}

    @app.get("/api/customers/{customer_id}/tickets")
    async def list_tickets(customer_id: str, on_behalf_of: OnBehalfOf = None) -> dict[str, Any]:
        tickets = backend.list_tickets(customer_id, _require_on_behalf_of(on_behalf_of))
        return {"customerId": customer_id, "tickets": tickets}

    @app.get("/api/orders/{order_id}/payments")
    async def get_payments(order_id: str, on_behalf_of: OnBehalfOf = None) -> dict[str, Any]:
        return backend.get_payments(order_id, _require_on_behalf_of(on_behalf_of))

    @app.get("/api/orders/{order_id}/shipments")
    async def get_shipments(order_id: str, on_behalf_of: OnBehalfOf = None) -> dict[str, Any]:
        return backend.get_shipments(order_id, _require_on_behalf_of(on_behalf_of))

    @app.get("/api/orders/{order_id}/refunds")
    async def get_refunds(order_id: str, on_behalf_of: OnBehalfOf = None) -> dict[str, Any]:
        return backend.get_refunds(order_id, _require_on_behalf_of(on_behalf_of))

    @app.post("/api/refunds/preview")
    async def preview_refund(body: RefundPreviewRequest, on_behalf_of: OnBehalfOf = None) -> dict[str, Any]:
        return backend.preview_refund(
            _require_on_behalf_of(on_behalf_of),
            order_id=body.order_id,
            amount_cents=body.amount_cents,
            reason_code=body.reason_code,
            line_ids=body.line_ids,
        )

    @app.post("/api/refunds")
    async def create_refund(
        body: RefundCreateRequest,
        on_behalf_of: OnBehalfOf = None,
        idempotency_key: IdempotencyKey = None,
    ) -> JSONResponse:
        refund, replayed = backend.create_refund(
            _require_on_behalf_of(on_behalf_of),
            idempotency_key=_require_idempotency_key(idempotency_key),
            order_id=body.order_id,
            amount_cents=body.amount_cents,
            reason_code=body.reason_code,
            line_ids=body.line_ids,
            case_id=body.case_id,
            approved_by=body.approved_by,
            note=body.note,
        )
        return _created(refund, replayed)

    @app.post("/api/escalations")
    async def create_escalation(
        body: EscalationCreateRequest,
        on_behalf_of: OnBehalfOf = None,
        idempotency_key: IdempotencyKey = None,
    ) -> JSONResponse:
        escalation, replayed = backend.create_escalation(
            _require_on_behalf_of(on_behalf_of),
            idempotency_key=_require_idempotency_key(idempotency_key),
            queue=body.queue,
            priority=body.priority,
            summary=body.summary,
            case_id=body.case_id,
            order_id=body.order_id,
        )
        return _created(escalation, replayed)

    # Test hooks for the evaluation harness. Not part of the real service contract.
    @app.post("/api/admin/reset")
    async def reset() -> dict[str, Any]:
        backend.reset()
        return {"status": "reset", "asOf": iso(backend.now())}

    @app.get("/api/admin/ledger")
    async def ledger() -> dict[str, Any]:
        return {"entries": backend.ledger}

    @app.get("/api/admin/access-log")
    async def access_log() -> dict[str, Any]:
        return {"entries": backend.access_log}

    return app


app = create_app()
