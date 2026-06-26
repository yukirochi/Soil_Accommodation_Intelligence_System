import sqlite3
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class temp_storage:
    
    def __init__(self, db_path: str, table_name: str):
        self.db_path = db_path
        self.table_name = table_name
        
        self.conn = sqlite3.connect(db_path)
        
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                atmospheric_temp REAL,
                soil_temp        REAL,
                humidity         REAL,
                rainfall         REAL,
                soil_moisture    REAL,
                grid_x           REAL NOT NULL,
                grid_y           REAL NOT NULL,
                timestamp        TEXT UNIQUE
            )
        """)
        
        self.conn.commit()
        
    def store(self, data_dict: dict) -> None:
        
        
        self.conn.execute(f"""
            INSERT OR IGNORE INTO {self.table_name} (
                 atmospheric_temp, soil_temp, humidity,
                 rainfall, soil_moisture, grid_x, grid_y, timestamp
             ) VALUES (
                :atmospheric_temp, :soil_temp, :humidity,
                :rainfall, :soil_moisture, :grid_x, :grid_y, :timestamp
                )
            """, data_dict)
        self.conn.commit()
        
    def flush(self):
        self.conn.execute(f'DROP TABLE IF EXISTS {self.table_name}')
        self.conn.commit()
      

    