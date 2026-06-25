import sqlite3

class temp_storage:
    
    def __init__(self, db_path: str, process_function):
        self.db_path = db_path
        self.process_function = process_function
        
        self.conn = sqlite3.connect(db_path)
        
        self._conn.execute("PRAGMA journal_mode=WAL")   # faster concurrent writes
        self._conn.execute("PRAGMA synchronous=NORMAL") # safe but not fsync-every-write
        self.conn.execute(""" 
                CREATE TABLE IF NOT EXISTS sensor_data (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                payload BLOB NOT NULL           -- raw JSON bytes, no TEXT overhead
                 )     
            """)
        
        self.conn.commit()
        
        
    