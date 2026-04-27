from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langsmith import traceable
from src.config import settings
from src.tools.order_query import query_order, query_user_orders
from src.tools.faq_search import search_faq
from src.tools.refund_checker import check_refund_eligibility

CS_SYSTEM_PROMPT = """You are a senior customer service agent for CloudOrder, an e-commerce platform.
For complex issues, follow this approach:
1. First understand the problem fully
2. Look up relevant order/account information using the available tools
3. Check policies if needed using the FAQ search
4. Reason through the situation step by step
5. Provide a clear resolution or next steps

Be empathetic, professional, and thorough. Always verify information before making conclusions.
If you cannot fully resolve an issue, clearly explain what you found and what the next steps are."""

_llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=settings.anthropic_api_key,
    max_tokens=1024,
)

_tools = [query_order, query_user_orders, search_faq, check_refund_eligibility]

_react_agent = create_react_agent(
    model=_llm,
    tools=_tools,
    prompt=CS_SYSTEM_PROMPT,
)

RECURSION_LIMIT = 10
FALLBACK_MESSAGE = (
    "I've been investigating your issue but it's more complex than I can resolve "
    "in this conversation. Let me summarize what I've found so far and connect you "
    "with a specialist who can help further."
)


@traceable(name="cs_react_agent")
async def run_cs_agent(query: str, user_id: str | None = None) -> str:
    user_context = f" (User ID: {user_id})" if user_id else ""
    full_query = f"{query}{user_context}"

    try:
        result = await _react_agent.ainvoke(
            {"messages": [("human", full_query)]},
            config={"recursion_limit": RECURSION_LIMIT},
        )
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
                return msg.content
        return FALLBACK_MESSAGE
    except Exception:
        return FALLBACK_MESSAGE
