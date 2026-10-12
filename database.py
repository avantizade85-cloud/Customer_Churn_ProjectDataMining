import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = BASE_DIR / "churn_predictions.db"


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def create_table():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            gender TEXT,
            senior_citizen INTEGER,
            partner TEXT,
            dependents TEXT,

            tenure REAL,
            phone_service TEXT,
            multiple_lines TEXT,
            internet_service TEXT,

            contract TEXT,
            payment_method TEXT,

            monthly_charges REAL,
            total_charges REAL,

            prediction TEXT,
            probability REAL,
            risk_level TEXT,

            recommendation TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()

    connection.close()


if __name__ == "__main__":

    create_table()

    print("Database created successfully!")

    print(f"Database location: {DATABASE_PATH}")