import sqlite3
import pandas as pd

fertile = pd.read_csv('datasets/soil_fertilization.csv')
moisture = pd.read_csv('datasets/soil_moist.csv')

moisture.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

moisture_df = moisture[['id','hour_sin','hour_cos','soil_temp','rain_fall',
                         'prev_5','prev_4','prev_3','prev_2','prev_1','future_moisture']]
fertile_df = fertile[fertile['Output'].isin([0, 1])]

conn = sqlite3.connect('permanent.db')
cur = conn.cursor()

# Drop tables if they already exist, so this is re-runnable
cur.execute('DROP TABLE IF EXISTS soil_moisture')
cur.execute('DROP TABLE IF EXISTS soil_fertility')

# Create soil_moisture with id as PRIMARY KEY
cur.execute('''
    CREATE TABLE soil_moisture (
        id INTEGER PRIMARY KEY,
        hour_sin REAL,
        hour_cos REAL,
        soil_temp REAL,
        rain_fall REAL,
        prev_5 REAL,
        prev_4 REAL,
        prev_3 REAL,
        prev_2 REAL,
        prev_1 REAL,
        future_moisture REAL
    )
''')

conn.commit()

# Now append the DataFrame data into the existing table
moisture_df.to_sql('soil_moisture', conn, if_exists='append', index=False)
fertile_df.to_sql('soil_fertility', conn, if_exists='replace', index=False)

conn.close()

# ---------------------------------------------------------------------------
# Temporary DB — same schema as permanent, but starts empty.
# Used by InputManager to store live inference rows without polluting
# the seeded training history in permanent.db.
# ---------------------------------------------------------------------------
temp_conn = sqlite3.connect('temporary.db')
temp_cur = temp_conn.cursor()

temp_cur.execute('DROP TABLE IF EXISTS soil_moisture')
temp_cur.execute('''
    CREATE TABLE soil_moisture (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hour_sin REAL,
        hour_cos REAL,
        soil_temp REAL,
        rain_fall REAL,
        prev_5 REAL,
        prev_4 REAL,
        prev_3 REAL,
        prev_2 REAL,
        prev_1 REAL,
        future_moisture REAL
    )
''')

temp_conn.commit()
temp_conn.close()

print("Databases initialised: permanent.db (seeded) + temporary.db (empty)")