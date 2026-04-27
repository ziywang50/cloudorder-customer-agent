import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx

from src.tools.order_query import query_order, query_user_orders
from src.tools.refund_checker import check_refund_eligibility


SAMPLE_ORDER = {
    "orderId": "ORD-123",
    "userId": 1,
    "status": "CONFIRMED",
    "totalAmount": 59.99,
    "buyerName": "John Doe",
    "buyerPhone": "555-0100",
    "buyerAddress": "123 Main St",
    "createdAt": "2026-04-20T10:00:00",
    "updatedAt": "2026-04-20T10:05:00",
    "items": [
        {
            "productId": 1,
            "productName": "Wireless Mouse",
            "productPrice": 29.99,
            "quantity": 2,
            "subtotal": 59.98,
        }
    ],
}


@pytest.fixture
def mock_httpx():
    with patch("src.tools.order_query.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        yield client


@pytest.fixture
def mock_httpx_refund():
    with patch("src.tools.refund_checker.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        yield client


async def test_query_order_success(mock_httpx):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = SAMPLE_ORDER
    resp.raise_for_status = MagicMock()
    mock_httpx.get = AsyncMock(return_value=resp)

    result = await query_order.ainvoke({"order_id": "ORD-123"})
    assert "ORD-123" in result
    assert "CONFIRMED" in result
    assert "Wireless Mouse" in result


async def test_query_order_not_found(mock_httpx):
    resp = MagicMock()
    resp.status_code = 404
    mock_httpx.get = AsyncMock(return_value=resp)

    result = await query_order.ainvoke({"order_id": "ORD-999"})
    assert "No order found" in result


async def test_query_order_timeout(mock_httpx):
    mock_httpx.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

    result = await query_order.ainvoke({"order_id": "ORD-123"})
    assert "timed out" in result


async def test_query_order_connection_error(mock_httpx):
    mock_httpx.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

    result = await query_order.ainvoke({"order_id": "ORD-123"})
    assert "Cannot reach" in result


async def test_query_user_orders_success(mock_httpx):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = [SAMPLE_ORDER]
    resp.raise_for_status = MagicMock()
    mock_httpx.get = AsyncMock(return_value=resp)

    result = await query_user_orders.ainvoke({"user_id": "1"})
    assert "1 order(s)" in result
    assert "ORD-123" in result


async def test_query_user_orders_empty(mock_httpx):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = []
    resp.raise_for_status = MagicMock()
    mock_httpx.get = AsyncMock(return_value=resp)

    result = await query_user_orders.ainvoke({"user_id": "999"})
    assert "No orders found" in result


async def test_refund_eligible(mock_httpx_refund):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = SAMPLE_ORDER
    resp.raise_for_status = MagicMock()
    mock_httpx_refund.get = AsyncMock(return_value=resp)

    result = await check_refund_eligibility.ainvoke({"order_id": "ORD-123"})
    assert "IS eligible" in result


async def test_refund_cancelled_order(mock_httpx_refund):
    cancelled = {**SAMPLE_ORDER, "status": "CANCELLED"}
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = cancelled
    resp.raise_for_status = MagicMock()
    mock_httpx_refund.get = AsyncMock(return_value=resp)

    result = await check_refund_eligibility.ainvoke({"order_id": "ORD-123"})
    assert "already cancelled" in result


async def test_refund_pending_order(mock_httpx_refund):
    pending = {**SAMPLE_ORDER, "status": "PENDING"}
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = pending
    resp.raise_for_status = MagicMock()
    mock_httpx_refund.get = AsyncMock(return_value=resp)

    result = await check_refund_eligibility.ainvoke({"order_id": "ORD-123"})
    assert "still pending" in result


async def test_refund_flash_sale_item(mock_httpx_refund):
    flash_order = {
        **SAMPLE_ORDER,
        "items": [
            {
                "productId": 1,
                "productName": "Flash Sale Headphones",
                "productPrice": 19.99,
                "quantity": 1,
                "subtotal": 19.99,
            }
        ],
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = flash_order
    resp.raise_for_status = MagicMock()
    mock_httpx_refund.get = AsyncMock(return_value=resp)

    result = await check_refund_eligibility.ainvoke({"order_id": "ORD-123"})
    assert "NOT eligible" in result
    assert "flash sale" in result


async def test_refund_not_found(mock_httpx_refund):
    resp = MagicMock()
    resp.status_code = 404
    mock_httpx_refund.get = AsyncMock(return_value=resp)

    result = await check_refund_eligibility.ainvoke({"order_id": "ORD-999"})
    assert "not found" in result
