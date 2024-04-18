import sqlite3
import pandas as pd

def load_df(db_path, table):
    """Load lung cancer data from SQLite database."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(table, con)
    con.close()
    return df
