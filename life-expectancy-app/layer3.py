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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class FinalPredictor:
    #This class loads data and trains the model ONCE upon initialization.
    #It then provides a method to make fast predictions and a helper to get data for web form dropdowns.

    def __init__(self):
  
        print("Initializing Engine: Loading data...")
        # imputers and encoders
        self.iter_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42) # iterative imputer for numerical features
        self.simple_imputer = SimpleImputer(strategy='most_frequent')                    # simple imputer for categorical features
        self.encoder = OneHotEncoder(handle_unknown='ignore')                            # one-hot encoder for categorical features

        # define class variables
        self.db_file = "life_expectancy_model_data.db"  # SQLite database file
        self.full_dataset = None            # Complete dataset made by combining all tables 
        self.ALL_FEATURES = None            # List of all feature column names
        self.X = None                             # Features for stacking model
        self.y_stack = None                       # stacking model target variable

        self.X_state_county_model_data = None                 # Features state_county_model feature data
        self.X_133_year_race_sex_model_data = None           # Features _133_year_race_sex_model feature data
        self.X_state_sex_model_data = None                    # Features state_sex_model feature data
        self.X_nchs_year_race_sex_model_data = None           # Features nchs_year_race_sex_model feature data
        self.X_ssa_year_sex_model_data = None                 # Features ssa_year_sex_model feature data

        self.y_state_county_model_data = None                 # Features state_county_model target data
        self.y_133_year_race_sex_model_data = None           # Features _133_year_race_sex_model target data
        self.y_state_sex_model_data = None                    # Features state_sex_model target data
        self.y_nchs_year_race_sex_model_data = None           # Features nchs_year_race_sex_model target data
        self.y_ssa_year_sex_model_data = None                 # Features ssa_year_sex_model target data

        self.X_stack_train = None                             # Features for stacking model training
        self.ssa_year_sex_model = None                          # model for ssa_year_sex
        self.state_sex_model = None                             # model for state_sex
        self.state_county_model = None                          # model for state_county
        self.nchs_year_race_sex_model = None                    # model for nchs_year_race_sex
        self._133_year_race_sex_model = None                    # model for _133_year_race_sex
        self.STACKING_MODEL = None                              # The final stacking model

        # --- Run all setup methods ---
        try:
            self._load_model_data()    # load prepared data
            self._build_models_from_pipelines()          # build pipelines for individual models
            self._train_models()          # train individual models
            self._build_stacking_model()    # build ensemble stacking model from individual models
            
            print("Training the Stacking Ensemble Model...")
            self._train_stacking_model()    # train the final stacking model
            print("Engine is trained and ready.")

        except sqlite3.OperationalError:
            print(f"CRITICAL ERROR: Database file '{self.db_file}' not found.")
            print("Please ensure the database is in the same directory.")
            raise
        except Exception as e:
            print(f"An error occurred during initialization: {e}")
            raise 
    
    def _load_model_data(self): # Load tables from SQLite database into DataFrames. 

        datasets = []
        conn = sqlite3.connect(self.db_file)        #connect to SQLite database
        datasets.append(pd.read_sql("SELECT * FROM state_county_le_p_censustract_model_data", conn))                #state_county_model_data
        datasets.append(pd.read_sql("SELECT * FROM _133_year_race_sex_model_data", conn))    #133_year_race_sex_model_data
        datasets.append(pd.read_sql("SELECT * FROM state_sex_le_p_stderr_model_data", conn))                      #state_sex_model_data
        datasets.append(pd.read_sql("SELECT * FROM nchs_year_race_sex_le_p_deathrate_model_data", conn))    #nchs_year_race_sex_model_data
        datasets.append(pd.read_sql("SELECT * FROM ssa_year_sex_model_data", conn))                #ssa_year_sex_model_data
        conn.close()

        for df in datasets:
            df.dropna(subset=['LifeExpectancy'], inplace=True)  #drop rows with missing LifeExpectancy

                                       #list for dataframes for all model tables

        self.full_dataset = pd.concat(datasets,ignore_index=True, sort=False)   #create master dataset by combining all model tables
        self.full_dataset = self.full_dataset.dropna(subset=['LifeExpectancy']) #drop rows with missing target variable

        self.y_state_county_model_data = datasets[0] ['LifeExpectancy']           # Target variable state_county_model_data
        self.y_133_year_race_sex_model_data = datasets[1]['LifeExpectancy']           # Target variable _133_year_race_sex_model_data
        self.y_state_sex_model_data = datasets[2]['LifeExpectancy']           # Target variable state_sex_model_data
        self.y_nchs_year_race_sex_model_data = datasets[3]['LifeExpectancy']     # Target variable nchs_year_race_sex_model_data
        self.y_ssa_year_sex_model_data = datasets[4]['LifeExpectancy']           # Target variable ssa_year_sex_model_data
        self.X_state_county_model_data = datasets[0].drop('LifeExpectancy', axis=1)           # Training features state_county_model_data
        self.X_133_year_race_sex_model_data = datasets[1].drop('LifeExpectancy', axis=1)     # Training features _133_year_race_sex_model_data
        self.X_state_sex_model_data = datasets[2].drop('LifeExpectancy', axis=1)              # Training features state_sex_model_data
        self.X_nchs_year_race_sex_model_data = datasets[3].drop('LifeExpectancy', axis=1)     # Training features nchs_year_race_sex_model_data
        self.X_ssa_year_sex_model_data = datasets[4].drop('LifeExpectancy', axis=1)           # Training features ssa_year_sex_model_data

        self.X = self.full_dataset.drop('LifeExpectancy', axis=1)       #obtain feature data
        self.y_stack = self.full_dataset['LifeExpectancy']         # obtain label data
        self.ALL_FEATURES = self.X.columns.tolist()                      #store feature names

    def _build_ssa_year_sex_pipeline(self):         # Builds the pipeline ssa_year_sex model.
        transformer = ColumnTransformer(transformers=[
            ('year', Pipeline([
                ('imputer', self.iter_imputer),
                ('scaler', StandardScaler())
            ]), ['Year']),
            ('sex', Pipeline([
                ('imputer', self.simple_imputer),
                ('encoder', self.encoder)
            ]), ['Sex'])
        ], remainder='drop')

        return Pipeline(steps=[('transformer', transformer), ('model', XGBRegressor(
                    n_estimators=100,learning_rate=0.1, max_depth=3, min_child_weight=3,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, n_jobs=-1))])

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
            ('stderr', Pipeline([
                ('imputer', self.iter_imputer),
                ('scaler', StandardScaler())
            ]), ['StandardError'])
        ], remainder='drop')

        return Pipeline(steps=[('transformer', transformer), ('model', XGBRegressor(
                    n_estimators=100,learning_rate=0.1, max_depth=3, min_child_weight=3,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, n_jobs=-1))])

    def _build_nchs_year_race_sex_pipeline(self):        #  Builds the pipeline for nchs_year_race_sex model.
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
            ('deathrate', Pipeline([
                ('imputer', self.iter_imputer),
                ('scaler', StandardScaler())
            ]), ['DeathRate'])            
        ], remainder='drop')

        return Pipeline(steps=[('transformer', transformer), ('model', XGBRegressor(
                    n_estimators=100,learning_rate=0.1, max_depth=3, min_child_weight=3,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, n_jobs=-1))])

    def _build_133_year_race_sex_pipeline(self):    #  Builds the pipeline for _133_year_race_sex model.
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
            ]), ['Race'])
        ], remainder='drop')

        return Pipeline(steps=[('transformer', transformer), ('model', XGBRegressor(
                    n_estimators=100,learning_rate=0.1, max_depth=3, min_child_weight=3,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, n_jobs=-1))])
    
    def _build_state_county_pipeline(self):         # Builds the pipeline for state_county model.
        transformer = ColumnTransformer(transformers=[
            ('state', Pipeline([
                ('imputer', self.simple_imputer),
                ('encoder', self.encoder)
            ]), ['State']),
            ('county', Pipeline([
                ('imputer', self.simple_imputer),
                ('encoder', self.encoder)
            ]), ['County']),
            ('censustract', Pipeline([
                ('imputer', self.iter_imputer),
                ('scaler', StandardScaler())
            ]), ['CensusTract'])
        ], remainder='drop')

        return Pipeline(steps=[('transformer', transformer), ('model', ElasticNet(
            alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=42))])  
    
    def _build_models_from_pipelines(self):          #Builds all individual pipelines. 
        self.ssa_year_sex_model = self._build_ssa_year_sex_pipeline()
        self.state_sex_model = self._build_state_sex_pipeline()
        self.state_county_model = self._build_state_county_pipeline()
        self.nchs_year_race_sex_model = self._build_nchs_year_race_sex_pipeline()
        self._133_year_race_sex_model = self._build_133_year_race_sex_pipeline()

    def _train_models(self):          #Trains all base models
        self.state_county_model.fit(self.X_state_county_model_data, self.y_state_county_model_data)
        self.ssa_year_sex_model.fit(self.X_ssa_year_sex_model_data, self.y_ssa_year_sex_model_data)  
        self.state_sex_model.fit(self.X_state_sex_model_data, self.y_state_sex_model_data)
        self._133_year_race_sex_model.fit(self.X_133_year_race_sex_model_data, self.y_133_year_race_sex_model_data)
        self.nchs_year_race_sex_model.fit(self.X_nchs_year_race_sex_model_data, self.y_nchs_year_race_sex_model_data)
            
    def _build_stacking_model(self): #   Builds the Meta_model using the individual pipelines and models.
        # Generate the out of fold meta-features for stacking model using full feature and target data
        preds_state_county_model_train = cross_val_predict(self.state_county_model, 
                                                           self.X_state_county_model_data, self.y_state_county_model_data, cv=5)
        preds_ssa_year_sex_model_train = cross_val_predict(self.ssa_year_sex_model, 
                                                           self.X_ssa_year_sex_model_data, self.y_ssa_year_sex_model_data, cv=5)
        preds_state_sex_model_train = cross_val_predict(self.state_sex_model, 
                                                        self.X_state_sex_model_data, self.y_state_sex_model_data, cv=5)
        preds_133_year_race_sex_model_train = cross_val_predict(self._133_year_race_sex_model, 
                                                                self.X_133_year_race_sex_model_data, self.y_133_year_race_sex_model_data, cv=5)        
        preds_nchs_year_race_sex_model_train = cross_val_predict(self.nchs_year_race_sex_model, 
                                                                 self.X_nchs_year_race_sex_model_data, self.y_nchs_year_race_sex_model_data, cv=5)
        #preds_state_county_model_train = cross_val_predict(self.state_county_model, self.X, self.y_stack, cv=5)
        #preds_ssa_year_sex_model_train = cross_val_predict(self.ssa_year_sex_model, self.X, self.y_stack, cv=5)
        #preds_state_sex_model_train = cross_val_predict(self.state_sex_model, self.X, self.y_stack, cv=5)
        #preds_nchs_year_race_sex_model_train = cross_val_predict(self.nchs_year_race_sex_model, self.X, self.y_stack, cv=5)
        #preds_133_year_race_sex_model_train = cross_val_predict(self._133_year_race_sex_model, self.X, self.y_stack, cv=5)
        #create stacking meta-model dataframe
        self.X_stack_train = pd.DataFrame({
            'state_county_model_pred': preds_state_county_model_train,
            'ssa_year_sex_model_pred': preds_ssa_year_sex_model_train,
            'state_sex_model_pred': preds_state_sex_model_train,
            'nchs_year_race_sex_model_pred': preds_nchs_year_race_sex_model_train,
            '_133_year_race_sex_model_pred': preds_133_year_race_sex_model_train
        })

        # Create Stacking Meta-Model. Using imputation just to make sure
        self.STACKING_MODEL = Pipeline([('imputer', self.iter_imputer),("model", LinearRegression())])     
   
    def _train_stacking_model(self):    #Trains the final stacking model on generated features against full dataset 
        self.STACKING_MODEL.fit(self.X_stack_train, self.y_stack)

    def make_prediction(self, state: str = None, county: str = None,
                        year: int = None, race: str = None, sex: str = None,
                       stderr: float = None, censustract: float = None, deathrate: float = None) -> float:
        # Create inference input dataframes for each individual model from user inputs
        ssa_year_sex_model_prediction_input = pd.DataFrame({'Year': [year], 'Sex': [sex]})        # input dataframe for ssa_year_sex_model
        state_sex_model_prediction_input = pd.DataFrame({'State': [state], 'Sex': [sex], 'StandardError': [stderr]})         # input dataframe for state_sex_model
        state_county_model_prediction_input = pd.DataFrame({'State': [state], 'County': [county], 'CensusTract': [censustract]})     # input dataframe for state_county_model
        nchs_year_race_sex_model_prediction_input = pd.DataFrame({'Year': [year], 'Race': [race], 'Sex': [sex], 'Deathrate': [deathrate]})  # input dataframe for nchs_year_race_sex_model
        _133_year_race_sex_model_prediction_input = pd.DataFrame({'Year': [year], 'Race': [race], 'Sex': [sex]})  # input dataframe for _133_year_race_sex_model
        # Generate predictions from individual models 
        ssa_year_sex_model_pred = self.ssa_year_sex_model.predict(ssa_year_sex_model_prediction_input)[0]                         
        state_sex_model_pred = self.state_sex_model.predict(state_sex_model_prediction_input)[0]                             
        state_county_model_pred = self.state_county_model.predict(state_county_model_prediction_input)[0]                          
        nchs_year_race_sex_model_pred = self.nchs_year_race_sex_model.predict(nchs_year_race_sex_model_prediction_input)[0]                    
        _133_year_race_sex_model_pred = self._133_year_race_sex_model.predict(_133_year_race_sex_model_prediction_input)[0] 

        # Create the stacking model input DataFrame
        stacking_model_input = pd.DataFrame({
            'state_county_model_pred': [state_county_model_pred],
            'ssa_year_sex_model_pred': [ssa_year_sex_model_pred],
            'state_sex_model_pred': [state_sex_model_pred],
            'nchs_year_race_sex_model_pred': [nchs_year_race_sex_model_pred],
            '_133_year_race_sex_model_pred': [_133_year_race_sex_model_pred]
        })

        # Make the final prediction using the stacking model
        final_prediction = self.STACKING_MODEL.predict(stacking_model_input)
        return final_prediction[0]
