"""Seed FAQ data into pgvector."""

import asyncio
import asyncpg
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import settings
from src.vectorstore.embeddings import embed_texts

FAQ_DATA = [
    # Returns
    {
        "question": "What is your return policy?",
        "answer": "We offer a 30-day return window from the date of delivery. Items must be in their original packaging and unused condition. To initiate a return, go to your order history and select 'Request Return'.",
        "category": "returns",
    },
    {
        "question": "Do I need the original packaging to return an item?",
        "answer": "Yes, items must be returned in their original packaging to qualify for a full refund. If the original packaging is damaged or missing, we may offer a partial refund at our discretion.",
        "category": "returns",
    },
    {
        "question": "Can I return flash sale items?",
        "answer": "No, all flash sale and lightning deal items are final sale and cannot be returned or refunded. Please review items carefully before purchasing during flash sales.",
        "category": "returns",
    },
    {
        "question": "How do I initiate a return?",
        "answer": "To start a return: 1) Log into your account, 2) Go to Order History, 3) Find the order and click 'Request Return', 4) Select the items and reason, 5) Print the return shipping label. Refunds are processed within 5-7 business days after we receive the item.",
        "category": "returns",
    },
    # Shipping
    {
        "question": "How long does standard shipping take?",
        "answer": "Standard shipping takes 5-7 business days and is free for orders over $50. For orders under $50, standard shipping costs $4.99. All orders require approximately 24 hours of processing time before shipping.",
        "category": "shipping",
    },
    {
        "question": "Do you offer express shipping?",
        "answer": "Yes, express shipping is available for $9.99 and delivers within 2-3 business days. Express orders placed before 2 PM are processed the same day.",
        "category": "shipping",
    },
    {
        "question": "How can I track my order?",
        "answer": "Once your order ships, you'll receive a tracking number via email. You can also check your order status anytime by logging into your account and viewing your order history.",
        "category": "shipping",
    },
    {
        "question": "Do you ship internationally?",
        "answer": "Currently, we only ship within the continental United States. We are working on expanding to international shipping in the future.",
        "category": "shipping",
    },
    # Payments
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept Visa, Mastercard, American Express, Discover, PayPal, and Apple Pay. All transactions are processed securely with SSL encryption.",
        "category": "payments",
    },
    {
        "question": "When is my credit card charged?",
        "answer": "Your credit card is authorized at the time of order placement but is only charged once the order is confirmed and ready to ship. If an order is cancelled before confirmation, no charge is applied.",
        "category": "payments",
    },
    {
        "question": "How does payment work for flash sales?",
        "answer": "For flash sales, your payment is authorized immediately when you place the order. You have a 15-minute confirmation window. If the order is confirmed within that window, the charge is finalized. If not, the authorization is released.",
        "category": "payments",
    },
    # Account
    {
        "question": "How do I create an account?",
        "answer": "Click 'Sign Up' on the top right of our website. You'll need to provide your email address, create a password, and verify your email. Account creation is free and gives you access to order tracking, wishlists, and exclusive deals.",
        "category": "account",
    },
    {
        "question": "How do I reset my password?",
        "answer": "Click 'Forgot Password' on the login page and enter your registered email. You'll receive a password reset link valid for 24 hours. If you don't receive the email, check your spam folder or contact support.",
        "category": "account",
    },
    {
        "question": "How can I view my order history?",
        "answer": "Log into your account and click on 'My Orders' in the navigation menu. You'll see all your past and current orders sorted by date, with options to view details, track shipments, or request returns.",
        "category": "account",
    },
    # Flash Sales
    {
        "question": "How do flash sales work?",
        "answer": "Flash sales offer limited-quantity items at steep discounts for a short time. Stock is limited and allocated first-come-first-served. Each user can purchase a maximum of one unit per flash sale item. Once stock runs out, the deal is over.",
        "category": "flash_sales",
    },
    {
        "question": "What happens if I don't confirm my flash sale order in time?",
        "answer": "Flash sale orders must be confirmed within 15 minutes of placement. If not confirmed in time, the order is automatically cancelled and the item is released back to available stock. Your payment authorization is also released.",
        "category": "flash_sales",
    },
    {
        "question": "Why was my flash sale order cancelled?",
        "answer": "Flash sale orders can be cancelled for several reasons: the 15-minute confirmation window expired, stock was depleted before confirmation, payment authorization failed, or you already purchased that item in the same flash sale event.",
        "category": "flash_sales",
    },
    {
        "question": "Can I buy more than one of a flash sale item?",
        "answer": "No, flash sales are limited to one unit per customer per item. This policy ensures fair access for all customers. Attempting to purchase multiple units will result in order cancellation.",
        "category": "flash_sales",
    },
]


async def seed():
    print(f"Connecting to {settings.database_url}...")
    conn = await asyncpg.connect(settings.database_url)

    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS faq_embeddings (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            category VARCHAR(100),
            embedding vector(384) NOT NULL
        );
        """
    )
    await conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_faq_embedding_hnsw
        ON faq_embeddings USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )

    existing = await conn.fetchval("SELECT COUNT(*) FROM faq_embeddings")
    if existing > 0:
        print(f"Table already has {existing} entries. Clearing and re-seeding...")
        await conn.execute("DELETE FROM faq_embeddings")

    questions = [faq["question"] for faq in FAQ_DATA]
    print(f"Generating embeddings for {len(questions)} FAQ entries...")
    embeddings = embed_texts(questions)

    for faq, embedding in zip(FAQ_DATA, embeddings):
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        await conn.execute(
            """
            INSERT INTO faq_embeddings (question, answer, category, embedding)
            VALUES ($1, $2, $3, $4::vector)
            """,
            faq["question"],
            faq["answer"],
            faq["category"],
            embedding_str,
        )

    count = await conn.fetchval("SELECT COUNT(*) FROM faq_embeddings")
    print(f"Seeded {count} FAQ entries successfully.")
    await conn.close()


if __name__ == "__main__":
    asyncio.run(seed())
