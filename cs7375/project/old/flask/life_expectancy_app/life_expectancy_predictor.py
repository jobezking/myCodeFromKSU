import pandas as pd
import numpy as np
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def life_expectancy_predictor_engine(geo_input: pd.DataFrame, demo_input: pd.DataFrame) -> float:
    """
    Canned example: compute a pseudo life expectancy from a single-row DataFrame.
    Returns a float.
    """
    db_file = "life_expectancy.db"
    table_geo = "life_expectancy_geography"
    table_demo = "life_expectancy_demography"
    # Load data from SQLite
    conn = sqlite3.connect(db_file)
    df_geo = pd.read_sql(f"SELECT * FROM {table_geo}", conn)
    df_demo = pd.read_sql(f"SELECT * FROM {table_demo}", conn)
    conn.close()
    
    df_geo['State_County'] = df_geo['State'] + '|' + df_geo['County']

    
    #prepare for training model
    X_geo = ['State_County', 'LifeExpectancyStandardError']
    y_geo = 'LifeExpectancy'
    
    geo_preprocessor = ColumnTransformer(transformers=[
        ('statecounty', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), ['State_County']),
        ('stderr', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['LifeExpectancyStandardError'])
    ])
    
    # build model for predicting life expectancy based on geographic data
    geo_model = Pipeline(steps=[
         ('preprocessor', geo_preprocessor),
         ('regressor', LinearRegression())
     ])
    
    # Train geographic life expectancy prediction model
    geo_model.fit(df_geo[X_geo], df_geo[y_geo])

    #Repeat process with demography data
    X_demo = ['Year', 'Race', 'Sex', 'AgeAdjustedDeathRate']
    y_demo   = 'LifeExpectancy'

    demo_preprocessor = ColumnTransformer(transformers=[
        ('year', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['Year']),
        ('race', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), ['Race']),
        ('sex', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), ['Sex']),
        ('deathrate', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['AgeAdjustedDeathRate'])
    ])

    # build model for predicting life expectancy based on demographic data
    demo_model = Pipeline(steps=[
        ('preprocessor', demo_preprocessor),
        ('regressor', LinearRegression())
    ])

    #Train demography model
    demo_model.fit(df_demo[X_demo], df_demo[y_demo])
    
    # Make predictions
    geo_pred = geo_model.predict(geo_input)
    demo_pred = demo_model.predict(demo_input)

    # Combine predictions (simple average for this example)
    return (geo_pred + demo_pred) / 2


def predict_life_expectancy(state: str, county: str, year: int, race: str, sex: str) -> float:
    """
    Receives inputs from Flask, builds a input DataFrames,
    calls compute_life_expectancy, and returns the float estimate.
    """

    state_county = f"{state}|{county}" #will use this construct again
    geo_input = pd.DataFrame([{
        'State_County': state_county,
        'LifeExpectancyStandardError': None
    }])
    demo_input = pd.DataFrame([{
        'Year': year,
        'Race': race,
        'Sex': sex,
        'AgeAdjustedDeathRate': None
    }]) 

    return life_expectancy_predictor_engine(geo_input, demo_input)

