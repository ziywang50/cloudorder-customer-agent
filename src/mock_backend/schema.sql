-- Durable state of the mock CloudOrder backend: everything the API can change.
-- Seed facts (customers, captures, shipments, tickets) stay in memory.
-- Portable across SQLite and Postgres. Money is integer cents. Timestamps are
-- ISO-8601 UTC strings from the backend clock (counters.clock, in seconds after AS_OF).

CREATE TABLE IF NOT EXISTS orders (
    order_id    TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS counters (
    name  TEXT PRIMARY KEY,
    value BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS refunds (
    refund_id              TEXT PRIMARY KEY,
    order_id               TEXT NOT NULL REFERENCES orders (order_id),
    customer_id            TEXT NOT NULL,
    amount_cents           BIGINT NOT NULL CHECK (amount_cents > 0),
    currency               TEXT NOT NULL,
    reason_code            TEXT NOT NULL,
    line_ids               TEXT NOT NULL,
    status                 TEXT NOT NULL,
    created_at             TEXT NOT NULL,
    expected_settlement_by TEXT NOT NULL,
    approved_by            TEXT NOT NULL,
    case_id                TEXT NOT NULL,
    note                   TEXT NOT NULL,
    capture_transaction_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_refunds_order ON refunds (order_id);

CREATE TABLE IF NOT EXISTS escalations (
    escalation_id         TEXT PRIMARY KEY,
    customer_id           TEXT NOT NULL,
    order_id              TEXT REFERENCES orders (order_id),
    queue                 TEXT NOT NULL,
    priority              TEXT NOT NULL,
    status                TEXT NOT NULL,
    summary               TEXT NOT NULL,
    case_id               TEXT NOT NULL,
    created_at            TEXT NOT NULL,
    first_response_due_by TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ledger_entries (
    entry_id    TEXT PRIMARY KEY,
    entry_type  TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    order_id    TEXT,
    payload     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_ledger_recorded ON ledger_entries (recorded_at);

-- One stored outcome per (customer, endpoint, key). Successful writes and business
-- rejections (for example REFUND_BLOCKED) are stored and replayed. Input validation
-- errors are never stored, so the client can fix the request and retry with the same key.
CREATE TABLE IF NOT EXISTS idempotency_records (
    customer_id     TEXT NOT NULL,
    endpoint        TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash    TEXT NOT NULL,
    status_code     INTEGER NOT NULL,
    response_body   TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    expires_at      TEXT NOT NULL,
    CONSTRAINT uq_idempotency_scope UNIQUE (customer_id, endpoint, idempotency_key)
);
