import pandas as pd
import sqlite3
import numpy as np

class InputManager:
    
    def __init__(self, db_path='database/permanent.db', temp_db_path='database/temporary.db'):
        self.db_path = db_path
        self.temp_db_path = temp_db_path
            
    def get_prediction(self, name, df):
        
        if df is None:
            return ValueError('df is empty')
        
        if name == 'soil_moisture':
            from model.soil_moisture_model.soil_moisture_model import soily
            df = self._clean_data(name, df)
            model = soily()
            return model.predict(df)
        elif name == 'soil_fertility':
            from model.soil_fertility_model.soil_fertility_model import ferti
            df = self._clean_data(name, df)
            model = ferti()
            return model.predict(df)
        else:
            raise ValueError(f"Model '{name}' not recognized.")
    
    def _clean_data(self, name,df):
        
        if name == 'soil_moisture':
            
            # input_format ['hour',soil_temp,rain_fall,soil_moisture]
            conn = sqlite3.connect(self.db_path)
            
            query = """ SELECT * FROM (SELECT * FROM soil_moisture WHERE future_moisture IS NOT NULL ORDER BY id DESC LIMIT 5) ORDER BY id ASC """
            
            last_5_col = pd.read_sql_query(query, conn)
            
            fill_future_moisture_query = """ 
            UPDATE soil_moisture
            SET future_moisture = ?
            WHERE id = (
            SELECT id FROM soil_moisture
            WHERE future_moisture IS NULL
            ORDER BY id DESC
            LIMIT 1
           )
            """
            conn.execute(fill_future_moisture_query, (float(df['soil_moisture'].iloc[-1]),))
            conn.commit()
            conn.close()   # ← fix: close after read + backfill, before feature engineering
            
            future_moisture_value = float(df['soil_moisture'].iloc[-1])
            
            df = pd.concat([last_5_col, df], ignore_index=True)
            
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['soil_temp'] = df['soil_temp'].round(2)
            df['rain_fall'] = df['rain_fall'].round(2)
            df['prev_5'] = df['future_moisture'].shift(5)
            df['prev_4'] = df['future_moisture'].shift(4)
            df['prev_3'] = df['future_moisture'].shift(3)
            df['prev_2'] = df['future_moisture'].shift(2)
            df['prev_1'] = df['future_moisture'].shift(1)
            
            input_data = df[['hour_sin','hour_cos','soil_temp','rain_fall','prev_5','prev_4','prev_3','prev_2','prev_1']].tail(1)
            input_data = input_data.astype({
                'hour_sin': np.float32,
                'hour_cos': np.float32,
                'soil_temp': np.float32,
                'rain_fall': np.float32,
                'prev_5': np.float32,
                'prev_4': np.float32,
                'prev_3': np.float32,
                'prev_2': np.float32,
                'prev_1': np.float32,
            })
            
            if input_data.isnull().values.any():
                print("Not enough history to build full feature row yet — skipping this cycle.")
                return ValueError('Not enough history to build full feature row yet — skipping this cycle.')
            
            latest = input_data.iloc[-1]
            self._store_to_permanent_db(latest, future_moisture_value)
            self._store_to_temporary_db(latest, future_moisture_value)
            return input_data
            
        
        elif name == 'soil_fertilization':

            # input_format: ['N','P','K','pH','EC','OC','S','Zn','Fe','Cu','Mn','B']
            FEATURE_COLS = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']

            conn = sqlite3.connect(self.db_path)

            missing = [c for c in FEATURE_COLS if c not in df.columns]
            if missing:
                conn.close()
                raise ValueError(f"soil_fertilization input missing columns: {missing}")

            input_data = df[FEATURE_COLS].tail(1).copy()
            input_data = input_data.astype({c: np.float32 for c in FEATURE_COLS})

            if input_data.isnull().values.any():
                print("soil_fertilization: incomplete input row — skipping this cycle.")
                conn.close()
                return None

            # log the raw reading for history/audit
            col_list = ', '.join(FEATURE_COLS)
            placeholders = ', '.join(['?'] * len(FEATURE_COLS))
            insert_query = f"""
                INSERT INTO soil_fertility ({col_list})
                VALUES ({placeholders})
            """
            latest = input_data.iloc[-1]
            conn.execute(insert_query, tuple(float(latest[c]) for c in FEATURE_COLS))
            conn.commit()
            conn.close()

            return input_data
        else:
            raise ValueError(f"Unrecognized clean_data target: '{name}'")

    def _store_moisture_row(self, db_path, latest, future_moisture_value):
        """
        Shared write logic for a soil_moisture row: backfills the previous
        row's future_moisture value, then inserts the new feature row.
        Used by both the permanent and temporary DB, so the two stay in sync.
        """
        conn = sqlite3.connect(db_path)
 
        fill_future_moisture_query = """
        UPDATE soil_moisture
        SET future_moisture = ?
        WHERE id = (
            SELECT id FROM soil_moisture
            WHERE future_moisture IS NULL
            ORDER BY id DESC
            LIMIT 1
        )
        """
        conn.execute(fill_future_moisture_query, (future_moisture_value,))
 
        insert_query = """
            INSERT INTO soil_moisture
                (hour_sin, hour_cos, soil_temp, rain_fall, prev_5, prev_4, prev_3, prev_2, prev_1, future_moisture)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
        """
        conn.execute(insert_query, (
            float(latest['hour_sin']),
            float(latest['hour_cos']),
            float(latest['soil_temp']),
            float(latest['rain_fall']),
            float(latest['prev_5']),
            float(latest['prev_4']),
            float(latest['prev_3']),
            float(latest['prev_2']),
            float(latest['prev_1']),
        ))
        conn.commit()
        conn.close()
 
    def _store_to_permanent_db(self, latest, future_moisture_value):
        self._store_moisture_row(self.db_path, latest, future_moisture_value)
 
    def _store_to_temporary_db(self, latest, future_moisture_value):
        self._store_moisture_row(self.temp_db_path, latest, future_moisture_value)
 