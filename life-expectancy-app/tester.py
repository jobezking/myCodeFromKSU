import pandas as pd
import sqlite3
from life_expectancy_predictor import Life_Expectancy_Predictor_Engine

if __name__ == "__main__":
    engine = Life_Expectancy_Predictor_Engine()
    # Example prediction
    prediction = engine.make_prediction(state="California", county="Los Angeles County", year=1950, race="White", sex="Female")
    print(f"Predicted Life Expectancy: {prediction:.2f} years") 

'''
def vet_database():

    db_file = "life_expectancy_model_data.db"

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # Fetch all the results
    tables = cursor.fetchall()
    for table in tables:
        # The result is a tuple, so we access the first element
        print(table[0])

    conn.close()
'''
