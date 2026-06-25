import duckdb
from pathlib import Path
import pandas as pd
import sqlite3


current_dir = Path(__file__).parent

database_path = current_dir.parent

sql_file_path = current_dir / 'sql' / 'soil_fertilization.sql'

conn = sqlite3.connect(database_path / 'database'/ 'soil.db')


with open(sql_file_path, 'r') as file:
    sql_query = file.read()

df = duckdb.query(sql_query).to_df()

df1 = df.copy()
df2 = df.copy()
df3 = df.copy()
df4 = df.copy()

df1['x'], df1['y'] = 0, 0
df2['x'], df2['y'] = 0, 1
df3['x'], df3['y'] = 1, 0
df4['x'], df4['y'] = 1, 1

df = pd.concat([df1, df2, df3, df4], ignore_index=True)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_sql('soil_fertilization', conn, if_exists='replace', index=False)



