import pytest
from unittest.mock import AsyncMock, patch, MagicMock


async def test_faq_agent_with_results():
    mock_results = [
        {
            "question": "What is your return policy?",
            "answer": "30-day return window.",
            "category": "returns",
            "similarity": 0.92,
        }
    ]
    with (
        patch("src.agents.faq_agent.search_faq", new_callable=AsyncMock, return_value=mock_results),
        patch("src.agents.faq_agent._llm") as mock_llm,
    ):
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Our return policy allows 30-day returns."))
        from src.agents.faq_agent import run_faq_agent

        result = await run_faq_agent("What is your return policy?")
        assert "return" in result.lower()


async def test_faq_agent_no_results():
    with patch("src.agents.faq_agent.search_faq", new_callable=AsyncMock, return_value=[]):
        from src.agents.faq_agent import run_faq_agent

        result = await run_faq_agent("How do I travel to Mars?")
        assert "couldn't find" in result.lower() or "rephrase" in result.lower()


async def test_order_agent_with_order_id():
    with (
        patch("src.agents.order_agent._llm") as mock_extract,
        patch("src.agents.order_agent.query_order") as mock_tool,
        patch("src.agents.order_agent._format_llm") as mock_format,
    ):
        mock_extract.ainvoke = AsyncMock(return_value=MagicMock(content="order:ORD-123"))
        mock_tool.ainvoke = AsyncMock(return_value="Order ID: ORD-123\nStatus: CONFIRMED")
        mock_format.ainvoke = AsyncMock(
            return_value=MagicMock(content="Your order ORD-123 is confirmed!")
        )

        from src.agents.order_agent import run_order_agent

        result = await run_order_agent("Where is my order ORD-123?")
        assert "ORD-123" in result


async def test_order_agent_no_id_no_user():
    with patch("src.agents.order_agent._llm") as mock_extract:
        mock_extract.ainvoke = AsyncMock(return_value=MagicMock(content="none"))

        from src.agents.order_agent import run_order_agent

        result = await run_order_agent("Where is my order?")
        assert "order ID" in result.lower() or "user ID" in result.lower()


async def test_cs_agent_returns_response():
    mock_result = {
        "messages": [MagicMock(content="I've investigated your issue. Here's what I found...")]
    }
    with patch("src.agents.cs_agent._react_agent") as mock_agent:
        mock_agent.ainvoke = AsyncMock(return_value=mock_result)

        from src.agents.cs_agent import run_cs_agent

        result = await run_cs_agent("I was charged twice for order ORD-456")
        assert "investigated" in result.lower() or len(result) > 0


async def test_cs_agent_fallback_on_error():
    with patch("src.agents.cs_agent._react_agent") as mock_agent:
        mock_agent.ainvoke = AsyncMock(side_effect=Exception("recursion limit"))

        from src.agents.cs_agent import run_cs_agent

        result = await run_cs_agent("Something very complex")
        assert "specialist" in result.lower() or "complex" in result.lower()
