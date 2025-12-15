import pandas as pd
import numpy as np
import sqlite3
#
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer #required for IterativeImputer import below
from sklearn.impute import SimpleImputer, IterativeImputer

def life_expectancy_predictor_engine(geo_input: pd.DataFrame, demo_input: pd.DataFrame) -> float:
    db_file = "life_expectancy.db"
    db_file_geo_full = "life_expectancy_geo_full.db"
    table_geo = "life_expectancy_geography"
    table_demo = "life_expectancy_demography"
    table_geo_full = "life_expectancy_geography_full"
    # Load data from SQLite
    conn1 = sqlite3.connect(db_file)
    conn2 = sqlite3.connect(db_file_geo_full)
    df_geo = pd.read_sql(f"SELECT * FROM {table_geo}", conn1)
    df_demo = pd.read_sql(f"SELECT * FROM {table_demo}", conn1)
    df_geo_full = pd.read_sql(f"SELECT * FROM {table_geo_full}", conn2)
    conn1.close()
    conn2.close()

    df_geo['State_County'] = df_geo['State'] + '|' + df_geo['County']
    df_geo_full['State_County_Census'] = df_geo_full['State'] + '|' + df_geo_full['County'] + '|' + df_geo_full['CensusTract'].astype(str)

    #prepare for training model
    X_geo = ['State_County', 'LifeExpectancyStandardError']
    y_geo = 'LifeExpectancy'
    #Repeat process with demography data
    X_demo = ['Year', 'Race', 'Sex', 'AgeAdjustedDeathRate']
    y_demo   = 'LifeExpectancy'
    #Use imputation to handle NaN's in y_geo_full
    numeric_cols = ["LifeExpectancy", "LifeExpectancyLow", "LifeExpectancyHigh", "LifeExpectancyStandardError"]
    imputer_y = IterativeImputer(estimator=BayesianRidge(), random_state=42)
    imputed = imputer_y.fit_transform(df_geo_full[numeric_cols])
    df_geo_full[numeric_cols] = imputed
    #repeat with full geography data
    X_geo_full = ['State_County_Census', 'LifeExpectancyStandardError','LifeExpectancyLow','LifeExpectancyHigh']
    y_geo_full = 'LifeExpectancy'

    # Define model-based imputers
    iter_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)

    geo_preprocessor_full = ColumnTransformer(transformers=[
        ('statecountycensus', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),   # categorical still needs simple imputer
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), ['State_County_Census']),
        ('lifeexpectlow', Pipeline([
            ('imputer', iter_imputer),
            ('scaler', StandardScaler())
        ]), ['LifeExpectancyLow']),
        ('lifeexpecthigh', Pipeline([
            ('imputer', iter_imputer),
            ('scaler', StandardScaler())
        ]), ['LifeExpectancyHigh']),
        ('stderr', Pipeline([
            ('imputer', iter_imputer),
            ('scaler', StandardScaler())
        ]), ['LifeExpectancyStandardError']),
    ])

    geo_preprocessor = ColumnTransformer(transformers=[
        ('statecounty', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), ['State_County']),
        ('stderr', Pipeline([
            ('imputer', iter_imputer),
            ('scaler', StandardScaler())
        ]), ['LifeExpectancyStandardError'])
    ])

    demo_preprocessor = ColumnTransformer(transformers=[
        ('year', Pipeline([
            ('imputer', iter_imputer),
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
            ('imputer', iter_imputer),
            ('scaler', StandardScaler())
        ]), ['AgeAdjustedDeathRate'])
    ])
    # build model for predicting life expectancy based on geographic data
    geo_model = Pipeline(steps=[
         ('preprocessor', geo_preprocessor),
         ('regressor', LinearRegression())
     ]) 
    # build model for predicting life expectancy based on demographic data
    demo_model = Pipeline(steps=[
        ('preprocessor', demo_preprocessor),
        ('regressor', LinearRegression())
    ])  
    # build model for predicting life expectancy based on full geographic data
    geo_model_full = Pipeline(steps=[
         ('preprocessor', geo_preprocessor_full),
         ('regressor', LinearRegression())
     ])
    
    # Train geographic life expectancy prediction model
    geo_model.fit(df_geo[X_geo], df_geo[y_geo])

    #Train demography model
    demo_model.fit(df_demo[X_demo], df_demo[y_demo])    

    #Train full geography model
    geo_model_full.fit(df_geo_full[X_geo_full], df_geo_full[y_geo_full])

    geo_full_input = pd.DataFrame([{
        'State_County_Census': None,
        'LifeExpectancyStandardError': None,
        'LifeExpectancyLow': None,
        'LifeExpectancyHigh': None
    }])    

    #Make predictions based on input data
    geo_pred = geo_model.predict(geo_input)
    demo_pred = demo_model.predict(demo_input)
    geo_full_pred = geo_model_full.predict(geo_full_input)  

    # Combine predictions (simple average for now)
    combined_pred = (geo_pred + demo_pred + geo_full_pred) / 3      
    return combined_pred

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