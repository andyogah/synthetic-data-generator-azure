from typing import List, Dict
import pandas as pd
import json

def load_data_from_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV, JSON, or Excel file.")

def load_data_from_database(connection_string: str, query: str) -> pd.DataFrame:
    import sqlalchemy
    engine = sqlalchemy.create_engine(connection_string)
    return pd.read_sql(query, engine)

def load_data(source: str, source_type: str, connection_string: str = None, query: str = None) -> pd.DataFrame:
    if source_type == 'file':
        return load_data_from_file(source)
    elif source_type == 'database':
        return load_data_from_database(connection_string, query)
    else:
        raise ValueError("Unsupported source type. Please provide 'file' or 'database'.")

def load_json_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as file:
        return json.load(file)