from src.vectorstore.connection import get_pool
from src.vectorstore.embeddings import embed_text


async def search_faq(query: str, top_k: int = 3) -> list[dict]:
    embedding = embed_text(query)
    pool = await get_pool()
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT question, answer, category,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM faq_embeddings
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            embedding_str,
            top_k,
        )

    results = []
    for row in rows:
        sim = float(row["similarity"])
        if sim < 0.3:
            continue
        results.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "category": row["category"],
                "similarity": round(sim, 4),
            }
        )
    return results
