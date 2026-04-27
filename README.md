# CloudOrder Customer Service Agent

Multi-agent customer service system for CloudOrder. Routes queries to specialized agents — **FAQ** (RAG), **Order Lookup** (direct tool call), or **CS Agent** (ReAct) — based on complexity.

## Architecture

```
User
 ↓
Router (LLM intent classifier — single fast call)
 ├── "faq"     → FAQ Agent (pgvector semantic search → LLM synthesis)
 ├── "order"   → Order Lookup (extract ID → query backend → format response)
 └── "complex" → CS Agent (ReAct: multi-step reasoning with tools)
```

**Why this design:** ReAct is powerful but expensive — each step requires an LLM call. Most customer queries are simple: "what's your return policy?" (FAQ lookup) or "where's my order?" (single API call). Only complex, ambiguous queries ("I was charged twice, check and help me fix it") benefit from multi-step reasoning. The router keeps cost and latency low by directing simple queries to lightweight agents and reserving the full ReAct loop for cases that actually need it.

### Agents

| Agent | Pattern | LLM Calls | When Used |
|-------|---------|-----------|-----------|
| FAQ | RAG (embed → search → synthesize) | 1 | Policy questions, how-to, general info |
| Order | Direct (extract → tool → format) | 2 | Order status, tracking, history |
| CS Agent | ReAct (reason → act → observe loop) | 3-10 | Disputes, refunds, multi-step problems |

### Tools Available to CS Agent

- `query_order` — Look up a single order by ID
- `query_user_orders` — List all orders for a user
- `search_faq` — Semantic search over FAQ knowledge base
- `check_refund_eligibility` — Check refund eligibility against business rules

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- Redis 7+

## Setup

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys and connection strings
```

Required environment variables:
- `ANTHROPIC_API_KEY` — Claude API key
- `DATABASE_URL` — PostgreSQL connection string (default: `postgresql://postgres:postgres@localhost:5432/customer_agent`)
- `REDIS_URL` — Redis connection string (default: `redis://localhost:6379/0`)
- `LANGCHAIN_API_KEY` — LangSmith API key (optional, for tracing)

### 3. Set up PostgreSQL with pgvector

```sql
CREATE DATABASE customer_agent;
\c customer_agent
CREATE EXTENSION vector;
```

### 4. Seed FAQ data

```bash
python scripts/seed_faq.py
```

### 5. Run the server

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## API

### `POST /chat`

```json
{
  "message": "What is your return policy?",
  "user_id": "123",
  "session_id": "optional-session-id"
}
```

Response:
```json
{
  "response": "We offer a 30-day return window...",
  "session_id": "abc-123",
  "intent": "faq"
}
```

### `POST /chat/stream`

Same request body, returns Server-Sent Events.

### `GET /health`

Returns `{"status": "ok"}`.

## Example Conversations

### 1. FAQ Path

```
User: "What's your return policy?"
→ Router classifies as "faq"
→ FAQ Agent embeds query, searches pgvector (top 3 results)
→ Synthesizes answer: "We offer a 30-day return window from delivery.
   Items must be in original packaging. Flash sale items are final sale."
```

### 2. Order Lookup Path

```
User: "Where's my order ORD-123?"
→ Router classifies as "order"
→ Order Agent extracts order ID, calls GET /api/orders/ORD-123
→ Formats response: "Your order ORD-123 is confirmed! It contains
   2x Wireless Mouse ($59.98 total), placed on April 20."
```

### 3. Complex / ReAct Path

```
User: "I was charged twice for order ORD-456, can I get a refund?"
→ Router classifies as "complex"
→ CS Agent (ReAct):
  Think: Need to verify the order and check charges
  Act: query_order("ORD-456") → sees order details, status CONFIRMED
  Think: Order exists, need to check refund eligibility
  Act: check_refund_eligibility("ORD-456") → eligible, within 30 days
  Think: Order is eligible for refund, not a flash sale item
  Respond: "I've verified order ORD-456 ($59.99, confirmed April 20).
   It is eligible for a refund — within the 30-day window with no
   flash sale items. I recommend contacting support to initiate
   the refund process for the duplicate charge."
```

## Testing

```bash
pytest tests/ -v
```

## Tracing

Traces are sent to LangSmith when `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` is set. View traces at [smith.langchain.com](https://smith.langchain.com) under the `cloudorder-customer-agent` project.
