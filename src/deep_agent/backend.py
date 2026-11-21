"""Typed async client for the CloudOrder backend.

Every call carries `X-On-Behalf-Of` for the authenticated customer, so the backend
enforces tenant scope on its side too. Transport errors and 5xx responses are retried
with backoff, but only for idempotent requests: GETs, the read-only refund preview, and
mutations that carry an `Idempotency-Key`. Retrying anything else could double a write.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from src.deep_agent.config import DeepAgentSettings


class BackendError(Exception):
    """A CloudOrder error response or an unreachable backend."""

    def __init__(self, status: int, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}

    @property
    def retryable(self) -> bool:
        return self.status >= 500


def _segment(value: str) -> str:
    """Escape a value for use as one URL path segment."""
    return quote(value, safe="")


class CloudOrderClient:
    def __init__(self, http: httpx.AsyncClient, *, max_attempts: int = 3, backoff_s: float = 0.2):
        self._http = http
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s

    @classmethod
    def in_process(cls, app: Any = None, *, timeout_s: float = 5.0, max_attempts: int = 3) -> CloudOrderClient:
        """Talk to the mock backend through ASGI, with no network or separate server."""
        from src.mock_backend.app import create_app

        transport = httpx.ASGITransport(app=app or create_app())
        http = httpx.AsyncClient(transport=transport, base_url="http://cloudorder.mock", timeout=timeout_s)
        return cls(http, max_attempts=max_attempts)

    @classmethod
    def remote(cls, base_url: str, *, timeout_s: float = 5.0, max_attempts: int = 3) -> CloudOrderClient:
        return cls(httpx.AsyncClient(base_url=base_url, timeout=timeout_s), max_attempts=max_attempts)

    @classmethod
    def from_settings(cls, settings: DeepAgentSettings) -> CloudOrderClient:
        if settings.backend_url:
            return cls.remote(settings.backend_url, timeout_s=settings.backend_timeout_s,
                              max_attempts=settings.backend_max_attempts)
        return cls.in_process(timeout_s=settings.backend_timeout_s,
                              max_attempts=settings.backend_max_attempts)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        customer_id: str,
        json: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        idempotent: bool | None = None,
    ) -> Any:
        headers = {"X-On-Behalf-Of": customer_id}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if idempotent is None:
            idempotent = method == "GET" or idempotency_key is not None
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts if idempotent else 1),
            wait=wait_exponential(multiplier=self._backoff_s, max=2.0),
            retry=retry_if_exception(lambda exc: isinstance(exc, BackendError) and exc.retryable),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                return await self._send(method, path, headers, json)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _send(self, method: str, path: str, headers: dict[str, str], json: dict[str, Any] | None) -> Any:
        try:
            response = await self._http.request(method, path, headers=headers, json=json)
        except httpx.TransportError as exc:
            raise BackendError(
                503, "BACKEND_UNAVAILABLE", f"CloudOrder did not respond ({type(exc).__name__})."
            ) from exc
        if response.status_code >= 400:
            try:
                body = response.json()
            except ValueError:
                body = {}
            if not isinstance(body, dict):
                body = {}
            fallback = "BACKEND_UNAVAILABLE" if response.status_code >= 500 else "BACKEND_ERROR"
            raise BackendError(
                response.status_code,
                body.get("errorCode", fallback),
                body.get("message", f"CloudOrder returned HTTP {response.status_code}."),
                details=body,
            )
        return response.json()

    # ---- reads ------------------------------------------------------------------------

    async def get_customer(self, customer_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/api/customers/{_segment(customer_id)}", customer_id=customer_id)

    async def list_orders(self, customer_id: str) -> list[dict[str, Any]]:
        body = await self._request("GET", f"/api/customers/{_segment(customer_id)}/orders",
                                   customer_id=customer_id)
        return body["orders"]

    async def list_tickets(self, customer_id: str) -> list[dict[str, Any]]:
        body = await self._request("GET", f"/api/customers/{_segment(customer_id)}/tickets",
                                   customer_id=customer_id)
        return body["tickets"]

    async def get_order(self, customer_id: str, order_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/api/orders/{_segment(order_id)}", customer_id=customer_id)

    async def get_payments(self, customer_id: str, order_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/api/orders/{_segment(order_id)}/payments",
                                   customer_id=customer_id)

    async def get_shipments(self, customer_id: str, order_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/api/orders/{_segment(order_id)}/shipments",
                                   customer_id=customer_id)

    async def get_refunds(self, customer_id: str, order_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/api/orders/{_segment(order_id)}/refunds",
                                   customer_id=customer_id)

    async def preview_refund(
        self, customer_id: str, *, order_id: str, amount_cents: int, reason_code: str,
        line_ids: list[str],
    ) -> dict[str, Any]:
        body = {"orderId": order_id, "amountCents": amount_cents, "reasonCode": reason_code,
                "lineIds": line_ids}
        return await self._request("POST", "/api/refunds/preview", customer_id=customer_id,
                                   json=body, idempotent=True)

    # ---- mutations (used by the approval flow) ------------------------------------------

    async def create_refund(
        self, customer_id: str, *, idempotency_key: str, order_id: str, amount_cents: int,
        reason_code: str, line_ids: list[str], case_id: str, approved_by: str, note: str = "",
    ) -> dict[str, Any]:
        body = {"orderId": order_id, "amountCents": amount_cents, "reasonCode": reason_code,
                "lineIds": line_ids, "caseId": case_id, "approvedBy": approved_by, "note": note}
        return await self._request("POST", "/api/refunds", customer_id=customer_id, json=body,
                                   idempotency_key=idempotency_key)

    async def create_escalation(
        self, customer_id: str, *, idempotency_key: str, queue: str, priority: str, summary: str,
        case_id: str, order_id: str | None = None,
    ) -> dict[str, Any]:
        body = {"queue": queue, "priority": priority, "summary": summary, "caseId": case_id,
                "orderId": order_id}
        return await self._request("POST", "/api/escalations", customer_id=customer_id, json=body,
                                   idempotency_key=idempotency_key)
