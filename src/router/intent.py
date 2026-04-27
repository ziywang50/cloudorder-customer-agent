from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langsmith import traceable
from src.config import settings

ROUTER_PROMPT = """Classify the user's intent into exactly one of:
- "faq": general knowledge questions (return policy, shipping, payments, account help, flash sale rules)
- "order": simple order lookup (user provides an order ID or asks about their orders)
- "complex": multi-step problems requiring investigation (disputes, double charges, refund requests with conditions, complaints needing multiple lookups)

Respond with ONLY the intent label, nothing else."""

_llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=settings.anthropic_api_key,
    max_tokens=10,
)

VALID_INTENTS = {"faq", "order", "complex"}


@traceable(name="classify_intent")
async def classify_intent(
    message: str, history: list[BaseMessage] | None = None
) -> str:
    context_messages = []
    if history:
        context_messages = history[-2:]

    messages = [
        SystemMessage(content=ROUTER_PROMPT),
        *context_messages,
        HumanMessage(content=message),
    ]

    resp = await _llm.ainvoke(messages)
    intent = resp.content.strip().lower().strip('"').strip("'")

    if intent not in VALID_INTENTS:
        return "complex"
    return intent
