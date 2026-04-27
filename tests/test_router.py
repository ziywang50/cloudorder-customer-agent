import pytest
from unittest.mock import AsyncMock, patch

from src.router.intent import classify_intent, VALID_INTENTS


@pytest.fixture
def mock_llm():
    with patch("src.router.intent._llm") as mock:
        mock.ainvoke = AsyncMock()
        yield mock


async def test_classify_faq(mock_llm):
    mock_llm.ainvoke.return_value = AsyncMock(content="faq")
    result = await classify_intent("What is your return policy?")
    assert result == "faq"


async def test_classify_order(mock_llm):
    mock_llm.ainvoke.return_value = AsyncMock(content="order")
    result = await classify_intent("Where is my order ORD-123?")
    assert result == "order"


async def test_classify_complex(mock_llm):
    mock_llm.ainvoke.return_value = AsyncMock(content="complex")
    result = await classify_intent("I was charged twice, can I get a refund?")
    assert result == "complex"


async def test_classify_invalid_defaults_to_complex(mock_llm):
    mock_llm.ainvoke.return_value = AsyncMock(content="unknown_intent")
    result = await classify_intent("Something weird")
    assert result == "complex"


async def test_classify_strips_quotes(mock_llm):
    mock_llm.ainvoke.return_value = AsyncMock(content='"faq"')
    result = await classify_intent("What payment methods do you accept?")
    assert result == "faq"


async def test_classify_with_history(mock_llm):
    from langchain_core.messages import HumanMessage, AIMessage

    mock_llm.ainvoke.return_value = AsyncMock(content="order")
    history = [
        HumanMessage(content="I need help"),
        AIMessage(content="Sure, how can I help?"),
    ]
    result = await classify_intent("Check order ORD-456", history)
    assert result == "order"
    call_args = mock_llm.ainvoke.call_args[0][0]
    assert len(call_args) == 5  # system + 2 history + human
