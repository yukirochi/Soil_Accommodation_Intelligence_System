import duckdb
import pandas as pd
import os

# Absolute path to the training/ directory where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sql_path = os.path.join(BASE_DIR, 'query.sql')
csv_path = os.path.join(BASE_DIR, 'uscrn_soil_2017_TX_hourly.csv')

with open(sql_path, 'r') as file:
    sql_query = file.read()

# Replace the relative CSV path in the SQL with the absolute one
sql_query = sql_query.replace(
    'training/uscrn_soil_2017_TX_hourly.csv',
    csv_path.replace('\\', '/')
)

df = duckdb.query(sql_query).to_df()
print(df.head())