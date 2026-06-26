import duckdb
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)


class cleaning:
    def __init__ (self, id:str):
        self.id = id
        
        
    def clean_input(self, input_data: dict) -> pd.DataFrame:
        
        sql_path = os.path.join(current_dir, 'cleaning_pipeline','sql', f"{self.id}.sql")
        
        data = pd.DataFrame([input_data])
        
        with open(sql_path, 'r') as file:
            sql_query = file.read()
        
        final_query = sql_query.format(inp = "data")
    
        clean_df = duckdb.query(final_query).to_df()
        
        return clean_df
        
