CREATE TABLE IF NOT EXISTS transactions (
    id               SERIAL PRIMARY KEY,
    transaction_id   VARCHAR(20) UNIQUE NOT NULL,
    transaction_amt  NUMERIC(12, 2),
    product_cd       VARCHAR(10),
    card4            VARCHAR(20),
    card6            VARCHAR(20),
    p_emaildomain    VARCHAR(100),
    transaction_dt   BIGINT,
    is_fraud_actual  INTEGER,
    is_fraud_predicted INTEGER,
    fraud_probability  NUMERIC(6, 4),
    processed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fraud_predicted
    ON transactions(is_fraud_predicted);

CREATE INDEX IF NOT EXISTS idx_processed_at
    ON transactions(processed_at);
