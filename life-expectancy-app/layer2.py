import pandas as pd
import sqlite3
import warnings

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # required
from sklearn.impute import SimpleImputer, IterativeImputer

from layer3 import FinalPredictor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class FeaturePredictor:
    def __init__(self):
  
        print("Initializing Engine: Loading data...")
        # imputers and encoders
        self.iter_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42) # iterative imputer for numerical features
        self.simple_imputer = SimpleImputer(strategy='most_frequent')                    # simple imputer for categorical features
        self.encoder = OneHotEncoder(handle_unknown='ignore')                            # one-hot encoder for categorical features

        # define class variables
        self.db_file = "life_expectancy_model_data.db"  # SQLite database file

        self.X_state_county_le_p_censustract_model_data = None                 # Features state_county_model feature data
        self.X_state_sex_le_p_stderr_model_data = None           # Features _133_year_race_sex_model feature data
        self.X_nchs_year_race_sex_le_p_deathrate_model_data  = None                    # Features state_sex_model feature data

        self.y_state_county_le_p_censustract_model_data= None                 # Features state_county_model feature data
        self.y_state_sex_le_p_stderr_model_data = None           # Features _133_year_race_sex_model feature data
        self.y_nchs_year_race_sex_le_p_deathrate_model_data  = None                    # Features state_sex_model feature data

        self.state_county_le_p_censustract_model = None                 # Features state_county_model feature data
        self.state_sex_le_p_stderr_model = None           # Features _133_year_race_sex_model feature data
        self.nchs_year_race_sex_le_p_deathrate_model  = None                    # Features state_sex_model feature data

        self.state_sex_model = None                             # model for state_sex
        self.state_county_model = None                          # model for state_county
        self.nchs_year_race_sex_model = None                    # model for nchs_year_race_sex

        # --- Run all setup methods ---
        try:
            self.load_model_data()    # load prepared data
            self.build_models_from_pipelines()          # build pipelines for individual models
            self.train_models()          # train individual models

        except sqlite3.OperationalError:
            print(f"CRITICAL ERROR: Database file '{self.db_file}' not found.")
            print("Please ensure the database is in the same directory.")
            raise
        except Exception as e:
            print(f"An error occurred during initialization: {e}")
            raise 
    
    def load_model_data(self): # Load tables from SQLite database into DataFrames. 

        conn = sqlite3.connect(self.db_file)        #connect to SQLite database
        state_county = pd.read_sql("SELECT * FROM state_county_le_p_censustract_model_data", conn)                #state_county_model_data
        state_sex = pd.read_sql("SELECT * FROM state_sex_le_p_stderr_model_data", conn)                #state_county_model_data
        nchs_year = pd.read_sql("SELECT * FROM nchs_year_race_sex_le_p_deathrate_model_data", conn)                #state_county_model_data
        conn.close()

        # Split features and target variables
        self.X_state_county_le_p_censustract_model_data = state_county.drop(columns=['CensusTract'])
        self.y_state_county_le_p_censustract_model_data = state_county['CensusTract']  
        self.X_state_sex_le_p_stderr_model_data = state_sex.drop(columns=['StandardError'])
        self.y_state_sex_le_p_stderr_model_data = state_sex['StandardError'] 
        self.X_nchs_year_race_sex_le_p_deathrate_model_data = nchs_year.drop(columns=['DeathRate'])
        self.y_nchs_year_race_sex_le_p_deathrate_model_data = nchs_year['DeathRate']

    def _build_state_sex_pipeline(self):        # Builds the pipeline for state_sex model.
        transformer = ColumnTransformer(transformers=[
            ('state', Pipeline([
                ('imputer', self.simple_imputer),
                ('encoder', self.encoder)
            ]), ['State']),
            ('sex', Pipeline([
                ('imputer', self.simple_imputer),
                ('encoder', self.encoder)
            ]), ['Sex']),
            ('lifeexpectancy', Pipeline([
                ('imputer', self.iter_imputer),
                ('scaler', StandardScaler())
            ]), ['LifeExpectancy'])
        ], remainder='drop')

        return Pipeline(steps=[('transformer', transformer), ('model', XGBRegressor(
                    n_estimators=100,learning_rate=0.1, max_depth=3, min_child_weight=3,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, n_jobs=-1))])

    def build_state_county_le_pipeline(self):         # Builds the pipeline for state_county model.
        transformer = ColumnTransformer(transformers=[
            ('state', Pipeline([
                ('imputer', self.simple_imputer),
                ('encoder', self.encoder)
            ]), ['State']),
            ('county', Pipeline([
                ('imputer', self.simple_imputer),
                ('encoder', self.encoder)
            ]), ['County']),
            ('lifeexpectancy', Pipeline([
                ('imputer', self.iter_imputer),
                ('scaler', StandardScaler())
            ]), ['LifeExpectancy'])
        ], remainder='drop')

        return Pipeline(steps=[('transformer', transformer), ('model', ElasticNet(
            alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=42))])

    def build_nchs_year_race_sex_le_pipeline(self):        #  Builds the pipeline for nchs_year_race_sex model.
        transformer = ColumnTransformer(transformers=[
            ('year', Pipeline([
                ('imputer', self.iter_imputer),
                ('scaler', StandardScaler())
            ]), ['Year']),
            ('sex', Pipeline([
                ('imputer', self.simple_imputer),
                ('encoder', self.encoder)
            ]), ['Sex']),
            ('race', Pipeline([
                ('imputer', self.simple_imputer),
                ('encoder', self.encoder)
            ]), ['Race']),
            ('lifeexpectancy', Pipeline([
                ('imputer', self.iter_imputer),
                ('scaler', StandardScaler())
            ]), ['LifeExpectancy'])
        ], remainder='drop')

        return Pipeline(steps=[('transformer', transformer), ('model', XGBRegressor(
                    n_estimators=100,learning_rate=0.1, max_depth=3, min_child_weight=3,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, n_jobs=-1))])
    
    def build_models_from_pipelines(self):             #Builds all individual pipelines. 
        self.state_county_le_model = self.build_state_county_le_pipeline()
        self.state_sex_le_model = self._build_state_sex_pipeline()
        self.nchs_year_race_sex_le_model = self.build_nchs_year_race_sex_le_pipeline()

    def train_models(self):            # Trains all individual models.
        print("Training state_county_le_p_censustract_model...")
        self.state_county_le_model.fit(self.X_state_county_le_p_censustract_model_data, self.y_state_county_le_p_censustract_model_data)
        print("state_county_le_p_censustract_model trained.")

        print("Training state_sex_le_p_stderr_model...")
        self.state_sex_le_model.fit(self.X_state_sex_le_p_stderr_model_data, self.y_state_sex_le_p_stderr_model_data)
        print("state_sex_le_p_stderr_model trained.")       

        print("Training nchs_year_race_sex_le_p_deathrate_model...")
        self.nchs_year_race_sex_le_model.fit(self.X_nchs_year_race_sex_le_p_deathrate_model_data, self.y_nchs_year_race_sex_le_p_deathrate_model_data)
        print("nchs_year_race_sex_le_p_deathrate_model trained.")

    def make_prediction(self, state: str = None, county: str = None, year: int = None, 
                        race: str = None, sex: str = None, le: float = None) -> float:
        # Create inference input dataframes for each individual model from user inputs
        state_sex_le_model_prediction_input = pd.DataFrame({'State': [state], 'Sex': [sex], 'LifeExpectancy' : [le]})         
        state_county_le_model_pred_prediction_input = pd.DataFrame({'State': [state], 'County': [county], 'LifeExpectancy' : [le]})        
        nchs_year_race_sex_le_model_prediction_input = pd.DataFrame({'Year': [year], 'Race': [race], 'Sex': [sex], 'LifeExpectancy' : [le]})      
        # Generate predictions from individual models 

        censustract = self.state_county_le_model.predict(state_county_le_model_pred_prediction_input)[0]                   
        stderr = self.state_sex_le_model.predict(state_sex_le_model_prediction_input)[0]                                                     
        deathrate = self.nchs_year_race_sex_le_model.predict(nchs_year_race_sex_le_model_prediction_input)[0] 
        
        final_prediction = FinalPredictor()  
        return final_prediction.make_prediction(state=state, county=county, year=year, race=race, sex=sex, 
                                                stderr=stderr, censustract=censustract, deathrate=deathrate)                   

