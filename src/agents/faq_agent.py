from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from src.config import settings
from src.vectorstore.retriever import search_faq

FAQ_SYSTEM_PROMPT = """You are a helpful customer service assistant for CloudOrder, an e-commerce platform.
Answer the customer's question based on the provided FAQ context.
Be friendly, concise, and accurate. If the context doesn't cover the question, say you don't have that information and suggest they contact support for further help."""

_llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=settings.anthropic_api_key,
    max_tokens=500,
)


@traceable(name="faq_agent")
async def run_faq_agent(query: str) -> str:
    results = await search_faq(query, top_k=3)

    if not results:
        return (
            "I couldn't find a specific answer to your question in our FAQ. "
            "Could you rephrase your question, or would you like me to connect you "
            "with a support representative for further help?"
        )

    context_parts = []
    for r in results:
        context_parts.append(f"Q: {r['question']}\nA: {r['answer']}")
    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=FAQ_SYSTEM_PROMPT),
        HumanMessage(
            content=f"FAQ Context:\n{context}\n\nCustomer question: {query}"
        ),
    ]

    resp = await _llm.ainvoke(messages)
    return resp.content
