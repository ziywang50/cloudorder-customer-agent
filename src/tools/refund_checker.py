from datetime import datetime, timedelta
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


FLASH_SALE_KEYWORDS = ["flash", "seckill", "limited", "lightning"]


@tool
@traceable(name="check_refund_eligibility")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception(_is_retryable),
)
async def check_refund_eligibility(order_id: str) -> str:
    """Check whether an order is eligible for a refund based on business rules: 30-day window, order status, and flash sale restrictions."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{settings.order_query_service_url}/api/orders/{order_id}"
            )
        if resp.status_code == 404:
            return f"Cannot check refund eligibility: order '{order_id}' not found."
        resp.raise_for_status()
    except httpx.TimeoutException:
        return "Cannot check refund eligibility: order service timed out."
    except httpx.ConnectError:
        return "Cannot check refund eligibility: order service unreachable."
    except httpx.HTTPStatusError:
        return "Cannot check refund eligibility: order service error."

    order = resp.json()
    reasons = []

    if order["status"] == "CANCELLED":
        return (
            f"Order {order_id} is already cancelled. "
            "No refund is needed as the order was not completed."
        )

    if order["status"] == "PENDING":
        return (
            f"Order {order_id} is still pending confirmation. "
            "It can be cancelled before confirmation without needing a refund."
        )

    created_str = order.get("createdAt", "")
    try:
        created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        days_since = (datetime.now(created.tzinfo) - created).days
    except (ValueError, TypeError):
        days_since = 0

    within_window = days_since <= 30

    is_flash_sale = False
    for item in order.get("items", []):
        name = (item.get("productName") or "").lower()
        if any(kw in name for kw in FLASH_SALE_KEYWORDS):
            is_flash_sale = True
            break

    if not within_window:
        reasons.append(
            f"Order was placed {days_since} days ago, exceeding the 30-day return window."
        )
    if is_flash_sale:
        reasons.append("Order contains flash sale items, which are final sale and non-refundable.")

    if reasons:
        return (
            f"Order {order_id} is NOT eligible for a refund.\n"
            f"Reasons:\n- " + "\n- ".join(reasons)
        )

    return (
        f"Order {order_id} IS eligible for a refund.\n"
        f"Status: {order['status']}\n"
        f"Total: ${order['totalAmount']:.2f}\n"
        f"Placed {days_since} day(s) ago (within 30-day window).\n"
        "No flash sale items detected.\n"
        "Next step: A customer service representative can initiate the refund process."
    )
