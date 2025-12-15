import sqlite3
import pandas as pd
import os

# Database file
db_file = "life_expectancy.db"

# CSV file paths
df_geo = pd.read_csv("life_expectancy_by_geography.csv", encoding='latin1')   # geography dataset
df_demo = pd.read_csv("life_expectancy_by_demography.csv", encoding='latin1') # demography dataset

# Table names
table_geo = "life_expectancy_geography"
table_demo = "life_expectancy_demography"

# Connect to SQLite (creates file if not exists) ---
conn = sqlite3.connect(db_file)

df_geo.to_sql(table_geo, conn, if_exists="replace", index=False)
df_demo.to_sql(table_demo, conn, if_exists="replace", index=False)

conn.close()