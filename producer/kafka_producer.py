import pandas as pd
import json
import time
from kafka import KafkaProducer
from dotenv import load_dotenv
import os

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
DELAY = 0.05
SAMPLE_SIZE = 10000

def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8")
    )

def load_data():
    print("Loading transaction data...")
    df = pd.read_csv(
        "data/train_transaction.csv",
        nrows=SAMPLE_SIZE,
        usecols=["TransactionID", "TransactionAmt", "ProductCD",
                 "card4", "card6", "P_emaildomain",
                 "isFraud", "TransactionDT"]
    )
    df = df.where(pd.notnull(df), None)
    print(f"Loaded {len(df):,} transactions")
    return df

def stream_transactions(producer, df):
    print(f"Streaming to Kafka topic: {KAFKA_TOPIC}")
    print("Press Ctrl+C to stop\n")

    for idx, row in df.iterrows():
        transaction = row.to_dict()
        transaction_id = str(transaction.get("TransactionID", idx))

        producer.send(
            KAFKA_TOPIC,
            key=transaction_id,
            value=transaction
        )

        is_fraud = transaction.get("isFraud", 0)
        amt = transaction.get("TransactionAmt", 0)
        flag = "FRAUD" if is_fraud == 1 else "OK   "
        print(f"[{flag}] TxID: {transaction_id} | Amount: ${amt:.2f}")

        time.sleep(DELAY)

        if idx % 500 == 0 and idx > 0:
            producer.flush()
            print(f"--- {idx:,} transactions sent ---")

def main():
    producer = create_producer()
    df = load_data()
    try:
        stream_transactions(producer, df)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        producer.flush()
        producer.close()
        print("Producer closed cleanly")

if __name__ == "__main__":
    main()
