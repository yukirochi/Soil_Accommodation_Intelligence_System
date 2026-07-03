import pandas as pd
import sqlite3
from model.soil_moisture_model.soil_moisture_model import soily


class Retraining:

    def __init__(self, temp_db_path='database/temporary.db', table_name='soil_moisture'):
        self.temp_db = temp_db_path
        self.table_name = table_name

    def train(self):

        model = soily()

        conn = sqlite3.connect(self.temp_db)
        query = """SELECT * FROM soil_moisture WHERE future_moisture IS NOT NULL"""

        df = pd.read_sql_query(query, conn)
        conn.close()                        

        if df.empty:
            print("Retraining skipped: no completed rows in temporary DB yet.")
            return False

        df.drop(columns='id', inplace=True, errors='ignore')

        try:
            result = model.train(df)         # raises on bad data; returns True on success
        except (TypeError, ValueError) as e:
            print(f"Retraining aborted: model.train() rejected the data — {e}")
            return False

        if result is True:
            self.flush()

        return result

    def flush(self):
        """Delete all rows with a filled future_moisture from the temporary DB."""
        conn = sqlite3.connect(self.temp_db)  
        query = """DELETE FROM soil_moisture WHERE future_moisture IS NOT NULL"""
        conn.execute(query)                   
        conn.commit()
        conn.close()
