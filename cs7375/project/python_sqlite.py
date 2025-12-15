import sqlite3
import pandas as pd
import os

## The SQLite database files will be part of the permanent Docker image.
#This is how to create the SQLite database files

# Database file
db_file = "life_expectancy.db"

# CSV file paths
csv_geo = "life_expectancy_by_geography.csv"   # geography dataset
csv_demo = "life_expectancy_by_demography.csv" # demography dataset

# Table names
table_geo = "life_expectancy_geography"
table_demo = "life_expectancy_demography"

# --- 2a. Connect to SQLite (creates file if not exists) ---
conn = sqlite3.connect(db_file)

# --- 2b. Write geography CSV to SQLite ---
df_geo = pd.read_csv(csv_geo)
df_geo.to_sql(table_geo, conn, if_exists="replace", index=False)

# --- 2c. Write demography CSV to SQLite ---
df_demo = pd.read_csv(csv_demo)
df_demo.to_sql(table_demo, conn, if_exists="replace", index=False)

# --- 2d. Disconnect ---
conn.close()

# --- 2e. Reconnect ---
conn = sqlite3.connect(db_file)

# --- 2f. Retrieve geography data into pandas ---
df_geo_from_db = pd.read_sql_query(f"SELECT * FROM {table_geo}", conn)

# --- 2g. Retrieve demography data into pandas ---
df_demo_from_db = pd.read_sql_query(f"SELECT * FROM {table_demo}", conn)

# --- 2h. Disconnect ---
conn.close()

# Confirm results
print("Geography DataFrame shape:", df_geo_from_db.shape)
print("Demography DataFrame shape:", df_demo_from_db.shape)

##delete database, which will be created in the same place as the script is run unless a path is specified
db_file = "life_expectancy.db"
if os.path.exists(db_file):
    os.remove(db_file)
    print("Database deleted.")
else:
    print("Database file not found."
