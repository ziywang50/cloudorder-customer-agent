import re
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from src.config import settings
from src.tools.order_query import query_order, query_user_orders

EXTRACT_PROMPT = """Extract the order ID or user ID from the customer's message.
Respond in exactly one of these formats:
order:<order_id>
user:<user_id>

If you find an order ID (like ORD-123, order number, etc.), respond with order:<id>.
If you find a user ID (numeric), respond with user:<id>.
If you can't find either, respond with: none"""

_llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=settings.anthropic_api_key,
    max_tokens=50,
)

_format_llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=settings.anthropic_api_key,
    max_tokens=500,
)


@traceable(name="order_agent")
async def run_order_agent(query: str, user_id: str | None = None) -> str:
    messages = [
        SystemMessage(content=EXTRACT_PROMPT),
        HumanMessage(content=query),
    ]
    resp = await _llm.ainvoke(messages)
    extracted = resp.content.strip().lower()

    tool_result: str
    if extracted.startswith("order:"):
        order_id = extracted.split(":", 1)[1].strip()
        tool_result = await query_order.ainvoke({"order_id": order_id})
    elif extracted.startswith("user:"):
        uid = extracted.split(":", 1)[1].strip()
        tool_result = await query_user_orders.ainvoke({"user_id": uid})
    elif user_id:
        tool_result = await query_user_orders.ainvoke({"user_id": user_id})
    else:
        return (
            "I'd be happy to help you look up your order! "
            "Could you provide your order ID (e.g., ORD-123) or your user ID?"
        )

    format_messages = [
        SystemMessage(
            content="You are a helpful customer service assistant. "
            "Format the order information below into a friendly, readable response for the customer."
        ),
        HumanMessage(
            content=f"Customer asked: {query}\n\nOrder data:\n{tool_result}"
        ),
    ]
    formatted = await _format_llm.ainvoke(format_messages)
    return formatted.content
