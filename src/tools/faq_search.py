from langchain_core.tools import tool
from langsmith import traceable
from src.vectorstore.retriever import search_faq as _search_faq


@tool
@traceable(name="search_faq")
async def search_faq(query: str) -> str:
    """Search the FAQ knowledge base for answers to customer questions about policies, shipping, payments, accounts, and flash sales."""
    results = await _search_faq(query, top_k=3)
    if not results:
        return "No relevant FAQ entries found for this query."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] ({r['category']}, relevance: {r['similarity']})\n"
            f"Q: {r['question']}\n"
            f"A: {r['answer']}"
        )
    return "\n\n".join(lines)
