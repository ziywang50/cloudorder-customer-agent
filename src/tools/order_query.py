import httpx
from langchain_core.tools import tool
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from src.config import settings


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500:
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    return False


def _format_order(order: dict) -> str:
    lines = [
        f"Order ID: {order['orderId']}",
        f"Status: {order['status']}",
        f"Total: ${order['totalAmount']:.2f}",
        f"Buyer: {order['buyerName']}",
        f"Phone: {order['buyerPhone']}",
        f"Address: {order['buyerAddress']}",
        f"Created: {order['createdAt']}",
        f"Updated: {order.get('updatedAt', 'N/A')}",
        "Items:",
    ]
    for item in order.get("items", []):
        lines.append(
            f"  - {item['productName']} x{item['quantity']} "
            f"@ ${item['productPrice']:.2f} = ${item['subtotal']:.2f}"
        )
    return "\n".join(lines)


@tool
@traceable(name="query_order")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception(_is_retryable),
)
async def query_order(order_id: str) -> str:
    """Look up a single order by its order ID. Returns order details including status, items, and buyer info."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{settings.order_query_service_url}/api/orders/{order_id}"
            )
        if resp.status_code == 404:
            return f"No order found with ID '{order_id}'."
        resp.raise_for_status()
        return _format_order(resp.json())
    except httpx.TimeoutException:
        return f"Order service timed out while looking up order '{order_id}'. Please try again later."
    except httpx.ConnectError:
        return f"Cannot reach the order service right now. Please try again later."
    except httpx.HTTPStatusError as e:
        if e.response.status_code >= 500:
            return f"Order service is experiencing issues. Please try again later."
        return f"Error looking up order '{order_id}': {e.response.status_code}"


@tool
@traceable(name="query_user_orders")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception(_is_retryable),
)
async def query_user_orders(user_id: str) -> str:
    """Look up all orders for a given user ID. Returns a summary list of the user's orders."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{settings.order_query_service_url}/api/orders/user/{user_id}"
            )
        resp.raise_for_status()
        orders = resp.json()
        if not orders:
            return f"No orders found for user '{user_id}'."
        lines = [f"Found {len(orders)} order(s) for user {user_id}:\n"]
        for o in orders:
            item_count = len(o.get("items", []))
            lines.append(
                f"  - {o['orderId']}  |  {o['status']}  |  "
                f"${o['totalAmount']:.2f}  |  {item_count} item(s)  |  {o['createdAt']}"
            )
        return "\n".join(lines)
    except httpx.TimeoutException:
        return f"Order service timed out while looking up orders for user '{user_id}'."
    except httpx.ConnectError:
        return "Cannot reach the order service right now. Please try again later."
    except httpx.HTTPStatusError as e:
        if e.response.status_code >= 500:
            return "Order service is experiencing issues. Please try again later."
        return f"Error looking up orders for user '{user_id}': {e.response.status_code}"
