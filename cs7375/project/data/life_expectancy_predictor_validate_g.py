import pandas as pd
import sqlite3
import warnings

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # required
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class Life_Expectancy_Predictor_Engine:
    #This class loads data and trains the model ONCE upon initialization.
    #It then provides a method to make fast predictions and a helper to get data for web form dropdowns.

    def __init__(self):
        # Initializes predictor engine. This constructor:
        # 1. Defines the preprocessors.
        # 2. Loads data from the specified SQLite database and splits data.
        # 3. Creates the master dataset.
        # 4. Trains and evaluates all base models.
        # 5. Trains the final StackingRegressor model.
  
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

        # Model names for evaluation metrics
        self.model_names = []                       # List of base model names
        self.X_train = []                   # List of Features training set (DataFrame)
        self.X_test = []                    # List of Features test set (DataFrame)
        self.y_train = []                   # List of Labels training set (Series)
        self.y_test = []                    # List of Labels test set (Series)
        self.mse = []                       # List of Mean-squared errors
        self.r2score = []                   # List of R2-scores
        self.rmse = []                      # List of root mean square errors

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
            self._load_model_data()    # load prepared data and split into train/test sets
            self._build_models_from_pipelines()          # build pipelines for individual models
            
            print("Training Base Models...")
            self._train_models()          # train individual models
            
            print("Evaluating Base Models...")
            self._evaluate_models()       # evaluate individual models
            
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
        #data was previously prepared when read from original CSV files and loaded into SQlite DB. Can simply be used.
        #ssa_year_sex_model_data        #SSA_Life_Expectancy_by_sex_1940_2001.csv and SSA_Life_Expectancy_by_sex_2002_2080.csv: SSA.gov
        #state_sex_model_data           #U.S._State_Life_Expectancy_by_Sex__2020.csv: Kaggle.com
        #state_county_model_data        #U.S._Life_Expectancy_at_Birth_by_State_and_Census_Tract_-_2010-2015.csv: CDC.gov
        #nchs_year_race_sex_model_data  #NCHS_-_Death_rates_and_life_expectancy_at_birth.csv: Kaggle.com
        #_133_year_race_sex_model_data  #133_life_expectancy_data.csv: data.gov

        datasets = []
        conn = sqlite3.connect(self.db_file)        #connect to SQLite database
        datasets.append(pd.read_sql("SELECT * FROM state_county_model_data", conn))                #state_county_model_data
        datasets.append(pd.read_sql("SELECT * FROM _133_year_race_sex_model_data", conn))    #133_year_race_sex_model_data
        datasets.append(pd.read_sql("SELECT * FROM state_sex_model_data", conn))                      #state_sex_model_data
        datasets.append(pd.read_sql("SELECT * FROM nchs_year_race_sex_model_data", conn))    #nchs_year_race_sex_model_data
        datasets.append(pd.read_sql("SELECT * FROM ssa_year_sex_model_data", conn))                #ssa_year_sex_model_data
        conn.close()
        
        # Names for models and evaluation tracking
        self.model_names = ['state_county_model', '_133_year_race_sex_model', 'state_sex_model', 'nchs_year_race_sex_model', 'ssa_year_sex_model']
        
        # Load and split each dataset
        for i, df in enumerate(datasets):
            df.dropna(subset=['LifeExpectancy'], inplace=True)  #drop rows with missing LifeExpectancy
            X, y = df.drop('LifeExpectancy', axis=1), df['LifeExpectancy']
            X_train, X_test, y_train, y_test = self._train_test_split(X, y)
            
            self.X_train.append(X_train)
            self.X_test.append(X_test)
            self.y_train.append(y_train)
            self.y_test.append(y_test)

        # Assign the first (state_county) set of splits back to the original class variables for clarity/compatibility with existing code structure
        self.X_state_county_model_data = self.X_train[0]
        self.y_state_county_model_data = self.y_train[0]
        self.X_133_year_race_sex_model_data = self.X_train[1]
        self.y_133_year_race_sex_model_data = self.y_train[1]
        self.X_state_sex_model_data = self.X_train[2]
        self.y_state_sex_model_data = self.y_train[2]
        self.X_nchs_year_race_sex_model_data = self.X_train[3]
        self.y_nchs_year_race_sex_model_data = self.y_train[3]
        self.X_ssa_year_sex_model_data = self.X_train[4]
        self.y_ssa_year_sex_model_data = self.y_train[4]

        self.full_dataset = pd.concat(datasets,ignore_index=True, sort=False)   #create master dataset by combining all model tables
        self.full_dataset = self.full_dataset.dropna(subset=['LifeExpectancy']) #drop rows with missing target variable

        self.X = self.full_dataset.drop('LifeExpectancy', axis=1)       #obtain feature data
        self.y_stack = self.full_dataset['LifeExpectancy']         # obtain label data
        self.ALL_FEATURES = self.X.columns.tolist()                      #store feature names

    def get_dropdown_options(self): #   Extracts unique, sorted lists for the form dropdowns and a map of states to their counties.
        if self.full_dataset is None:
            return {}

        try:        # Get flat lists
            states_list = sorted(self.full_dataset['State'].dropna().unique())
            years_list0 = sorted(self.full_dataset['Year'].dropna().unique())
            years_list = [int(year) for year in years_list0 if not pd.isna(year)] # Convert floats to ints for dropdown, skipping NaN
            races_list = sorted(self.full_dataset['Race'].dropna().unique())
            sexes_list = sorted(self.full_dataset['Sex'].dropna().unique())

            # --- Create the State -> County Map ---
            # 1. Get only 'State' and 'County' columns.
            # 2. Drop rows where either is missing.
            # 3. Drop duplicate pairs to get unique (State, County) combinations.
            state_county_pairs = self.full_dataset[['State', 'County']].dropna().drop_duplicates()
            
            # 4. Group by 'State' and aggregate the 'County' values into a sorted list.
            # 5. Convert the result to a dictionary.
            state_county_map = (state_county_pairs.groupby('State')['County'].apply(lambda x: sorted(list(x))).to_dict())

            options = {
                'states': states_list,
                'years': years_list,
                'races': races_list,
                'sexes': sexes_list,
                'state_county_map': state_county_map  # Add the new map
            }
            return options
        except Exception as e:
            print(f"Error getting dropdown options: {e}")
            return {'states': [], 'years': [], 'races': [], 'sexes': [], 'state_county_map': {}}

    def _train_test_split(self, X, y):
        # Splits features X and labels y into training and testing sets (80/20 split)
        return train_test_split(X, y, test_size=0.2, random_state=42)

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

        return Pipeline(steps=[('transformer', transformer),('model', LinearRegression())])

    def _build_state_sex_pipeline(self):        # Builds the pipeline for state_sex model.
        transformer = ColumnTransformer(transformers=[
            ('state', Pipeline([
                ('imputer', self.simple_imputer),
                ('encoder', self.encoder)
            ]), ['State']),
            ('sex', Pipeline([
                ('imputer', self.simple_imputer),
                ('encoder', self.encoder)
            ]), ['Sex'])
        ], remainder='drop')

        return Pipeline(steps=[('transformer', transformer), ('model', LinearRegression())])

    def _build_state_county_pipeline(self):         # Builds the pipeline for state_county model.
        transformer = ColumnTransformer(transformers=[
            ('state', Pipeline([
                ('imputer', self.simple_imputer),
                ('encoder', self.encoder)
            ]), ['State']),
            ('county', Pipeline([
                ('imputer', self.simple_imputer),
                ('encoder', self.encoder)
            ]), ['County'])
        ], remainder='drop')

        return Pipeline(steps=[('transformer', transformer), ('model', LinearRegression())])

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
            ]), ['Race'])
        ], remainder='drop')

        return Pipeline(steps=[('transformer', transformer), ('model', LinearRegression())])

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

        return Pipeline(steps=[('transformer', transformer), ('model', LinearRegression())])
    
    def _build_models_from_pipelines(self):          #Builds all individual pipelines. 
        self.ssa_year_sex_model = self._build_ssa_year_sex_pipeline()
        self.state_sex_model = self._build_state_sex_pipeline()
        self.state_county_model = self._build_state_county_pipeline()
        self.nchs_year_race_sex_model = self._build_nchs_year_race_sex_pipeline()
        self._133_year_race_sex_model = self._build_133_year_race_sex_pipeline()
        # Create a list of all base models for easier iteration in evaluation
        self.base_models = [
            self.state_county_model,
            self._133_year_race_sex_model,
            self.state_sex_model,
            self.nchs_year_race_sex_model,
            self.ssa_year_sex_model
        ]

    def _train_models(self):          #Trains all base models on their respective training sets
        self.state_county_model.fit(self.X_state_county_model_data, self.y_state_county_model_data)
        self._133_year_race_sex_model.fit(self.X_133_year_race_sex_model_data, self.y_133_year_race_sex_model_data)
        self.state_sex_model.fit(self.X_state_sex_model_data, self.y_state_sex_model_data)
        self.nchs_year_race_sex_model.fit(self.X_nchs_year_race_sex_model_data, self.y_nchs_year_race_sex_model_data)
        self.ssa_year_sex_model.fit(self.X_ssa_year_sex_model_data, self.y_ssa_year_sex_model_data)  
    
    def _evaluate_models(self):         #Evaluates all base models on their respective test sets
        self.mse = []
        self.r2score = []
        self.rmse = []
        
        for i, model in enumerate(self.base_models):
            X_test_set = self.X_test[i]
            y_test_set = self.y_test[i]
            
            y_pred = model.predict(X_test_set)
            
            # Calculate metrics
            current_mse = mean_squared_error(y_test_set, y_pred)
            current_r2 = r2_score(y_test_set, y_pred)
            current_rmse = current_mse ** 0.5
            
            # Store results
            self.mse.append(current_mse)
            self.r2score.append(current_r2)
            self.rmse.append(current_rmse)
            
            print(f"  {self.model_names[i]:<25}: MSE={current_mse:.4f}, R2={current_r2:.4f}, RMSE={current_rmse:.4f}")

    def _build_stacking_model(self): #   Builds the Meta_model using the individual pipelines and models.
        # Generate the out of fold meta-features for stacking model using full feature and target data
        preds_state_county_model_train = cross_val_predict(self.state_county_model, self.X, self.y_stack, cv=5, n_jobs=-1)
        preds_ssa_year_sex_model_train = cross_val_predict(self.ssa_year_sex_model, self.X, self.y_stack, cv=5, n_jobs=-1)
        preds_state_sex_model_train = cross_val_predict(self.state_sex_model, self.X, self.y_stack, cv=5, n_jobs=-1)
        preds_nchs_year_race_sex_model_train = cross_val_predict(self.nchs_year_race_sex_model, self.X, self.y_stack, cv=5, n_jobs=-1)
        preds_133_year_race_sex_model_train = cross_val_predict(self._133_year_race_sex_model, self.X, self.y_stack, cv=5, n_jobs=-1)
        
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
        
        # Evaluate Stacking Model on its training set (the meta-features)
        y_stack_pred = self.STACKING_MODEL.predict(self.X_stack_train)
        stack_mse = mean_squared_error(self.y_stack, y_stack_pred)
        stack_r2 = r2_score(self.y_stack, y_stack_pred)
        stack_rmse = stack_mse ** 0.5
        
        # Store for plotting and display
        self.model_names.append('STACKING_MODEL (Train)')
        self.mse.append(stack_mse)
        self.r2score.append(stack_r2)
        self.rmse.append(stack_rmse)
        
        print(f"  STACKING_MODEL (Train)      : MSE={stack_mse:.4f}, R2={stack_r2:.4f}, RMSE={stack_rmse:.4f}")
        

    def make_prediction(self, state: str = None, county: str = None,
                        year: int = None, race: str = None, sex: str = None) -> float:
        # Create inference input dataframes for each individual model from user inputs
        ssa_year_sex_model_prediction_input = pd.DataFrame({'Year': [year], 'Sex': [sex]})        # input dataframe for ssa_year_sex_model
        state_sex_model_prediction_input = pd.DataFrame({'State': [state], 'Sex': [sex]})         # input dataframe for state_sex_model
        state_county_model_prediction_input = pd.DataFrame({'State': [state], 'County': [county]})     # input dataframe for state_county_model
        nchs_year_race_sex_model_prediction_input = pd.DataFrame({'Year': [year], 'Race': [race], 'Sex': [sex]})  # input dataframe for nchs_year_race_sex_model
        _133_year_race_sex_model_prediction_input = pd.DataFrame({'Year': [year], 'Race': [race], 'Sex': [sex]})  # input dataframe for _133_year_race_sex_model
        
        # Generate predictions from individual models 
        # Note: If an input is NaN/None, the pipeline's imputer/encoder will handle it.
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

    def plot_model_performance(self):
        # Create a DataFrame for plotting
        performance_df = pd.DataFrame({
            'Model': self.model_names,
            'MSE': self.mse,
            'R2 Score': self.r2score,
            'RMSE': self.rmse
        })

        # --- Plot 1: R2 Score (Measure of fit, higher is better) ---
        plt.figure(figsize=(12, 6))
        sns.barplot(x='R2 Score', y='Model', data=performance_df, palette='viridis')
        plt.title('R-squared ($R^2$) Score for Each Model (Higher is Better) ')
        plt.xlabel('R-squared Score ($R^2$)')
        plt.ylabel('Model Name')
        plt.xlim(0.0, 1.0)
        plt.grid(axis='x', linestyle='--')
        plt.tight_layout()
        plt.show()

        # --- Plot 2: Mean Squared Error (Measure of error, lower is better) ---
        plt.figure(figsize=(12, 6))
        sns.barplot(x='MSE', y='Model', data=performance_df, palette='magma')
        plt.title('Mean Squared Error (MSE) for Each Model (Lower is Better) ')
        plt.xlabel('Mean Squared Error (MSE)')
        plt.ylabel('Model Name')
        plt.grid(axis='x', linestyle='--')
        plt.tight_layout()
        plt.show()

        print("\n--- Model Performance Summary ---")
        print(performance_df)


if __name__ == "__main__":
    engine = Life_Expectancy_Predictor_Engine()
    
    # Example prediction
    print("\n--- Example Prediction ---")
    prediction = engine.make_prediction(state="California", county="Los Angeles County", year=1950, race="White", sex="Female")
    print(f"Predicted Life Expectancy: {prediction:.2f} years") 
    
    # Visualize and print performance metrics
    engine.plot_model_performance()