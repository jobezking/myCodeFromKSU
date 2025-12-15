import pandas as pd
import sqlite3
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # required
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import StackingRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class LifeExpectancyPredictorEngine:
    """
    This class loads data and trains the model ONCE upon initialization.
    It then provides a method to make fast predictions and a helper
    to get data for web form dropdowns.
    """

    def __init__(self):
        """
        Initializes the predictor engine.
        This constructor:
        1. Defines the preprocessors.
        2. Loads data from the specified SQLite database.
        3. Creates the master dataset.
        4. Trains the final StackingRegressor model.
        """
        print("Initializing Engine: Loading data...")
        # imputers and encoders
        self.iter_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)
        self.simple_imputer = SimpleImputer(strategy='most_frequent')
        self.encoder = OneHotEncoder(handle_unknown='ignore')

        # database file
        self.db_file = "life_expectancy_tables.db"

        # placeholders
        self.full_dataset = None
        self.ALL_FEATURES = None
        self.STACKING_MODEL = None
        self.X_train = None
        self.y_train = None

        # --- Run all setup methods ---
        try:
            # load and prepare data
            self._load_data()
            self._prepare_dataset()
            self._train_test_split()

            # build pipelines
            self.sex_year_pipeline = self._build_sex_year_pipeline()
            self.sex_state_pipeline = self._build_sex_state_pipeline()
            self.state_county_pipeline = self._build_state_county_pipeline()
            self.race_sex_pipeline = self._build_race_sex_pipeline()

            # build and train ensemble
            self._build_stacking_model()
            
            print("Training the Stacking Ensemble Model...")
            self._train_stacking_model()
            print("Engine is trained and ready.")

        except sqlite3.OperationalError:
            print(f"CRITICAL ERROR: Database file '{self.db_file}' not found.")
            print("Please ensure the database is in the same directory.")
            raise
        except Exception as e:
            print(f"An error occurred during initialization: {e}")
            raise

    def _load_data(self):
        """Load tables from SQLite database into DataFrames."""
        conn = sqlite3.connect(self.db_file)
        datasets = []
        datasets.append(pd.read_sql("SELECT * FROM ssa_life_expectancy_sex_year", conn))
        datasets.append(pd.read_sql("SELECT * FROM life_expectancy_sex_state", conn))
        datasets.append(pd.read_sql("SELECT * FROM life_expectancy_state_county", conn))
        datasets.append(pd.read_sql("SELECT * FROM life_expectancy_race_sex", conn))
        datasets.append(pd.read_sql("SELECT * FROM life_expectancy_race_sex_year", conn))
        conn.close()

        self.full_dataset = pd.concat(datasets,ignore_index=True, sort=False)

    def _prepare_dataset(self):
        """Clean dataset and drop rows without target."""
        self.full_dataset = self.full_dataset.dropna(subset=['LifeExpectancy'])
        # Ensure 'Year' is numeric for sorting, handle potential errors
        self.full_dataset['Year'] = pd.to_numeric(self.full_dataset['Year'], errors='coerce')
        self.full_dataset = self.full_dataset.dropna(subset=['Year'])
        self.full_dataset['Year'] = self.full_dataset['Year'].astype(int)


    def _train_test_split(self):
        """Split dataset into train/test sets."""
        X = self.full_dataset.drop('LifeExpectancy', axis=1)
        y = self.full_dataset['LifeExpectancy']
        self.ALL_FEATURES = X.columns.tolist()
        # We'll use the full dataset for training the final production model
        self.X_train = X
        self.y_train = y

    def _build_sex_year_pipeline(self):
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

        return Pipeline(steps=[
            ('transformer', transformer),
            ('model', LinearRegression())
        ])

    def _build_sex_state_pipeline(self):
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

        return Pipeline(steps=[
            ('transformer', transformer),
            ('model', LinearRegression())
        ])

    def _build_state_county_pipeline(self):
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

        return Pipeline(steps=[
            ('transformer', transformer),
            ('model', LinearRegression())
        ])

    def _build_race_sex_pipeline(self):
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
            ]), ['AgeAdjustedDeathRate'])
        ], remainder='drop')

        return Pipeline(steps=[
            ('transformer', transformer),
            ('model', LinearRegression())
        ])

        def _build_race_sex_year_pipeline(self):
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

        return Pipeline(steps=[
            ('transformer', transformer),
            ('model', LinearRegression())
        ])

    def _build_stacking_model(self):
        estimators = [
            ('sex_year_model', self.sex_year_pipeline),
            ('sex_state_model', self.sex_state_pipeline),
            ('state_county_model', self.state_county_pipeline),
            ('race_sex_model', self.race_sex_pipeline),
            ('race_sex_year_model', self.race_sex_year_pipeline)
        ]
        self.STACKING_MODEL = StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),
            cv=5, # Use cross-validation on the training data
            passthrough=False
        )

    def _train_stacking_model(self):
        """Trains the final stacking model on the full dataset."""
        self.STACKING_MODEL.fit(self.X_train, self.y_train)

    def get_dropdown_options(self):
        """
        Extracts unique, sorted lists for the form dropdowns
        and a map of states to their counties.
        """
        if self.full_dataset is None:
            return {}

        try:
            # Get flat lists
            states_list = sorted(self.full_dataset['State'].dropna().unique())
            years_list = sorted(self.full_dataset['Year'].dropna().unique(), reverse=True)
            races_list = sorted(self.full_dataset['Race'].dropna().unique())
            sexes_list = sorted(self.full_dataset['Sex'].dropna().unique())

            # --- Create the State -> County Map ---
            # 1. Get only 'State' and 'County' columns.
            # 2. Drop rows where either is missing.
            # 3. Drop duplicate pairs to get unique (State, County) combinations.
            state_county_pairs = self.full_dataset[['State', 'County']].dropna().drop_duplicates()
            
            # 4. Group by 'State' and aggregate the 'County' values into a sorted list.
            # 5. Convert the result to a dictionary.
            state_county_map = (
                state_county_pairs.groupby('State')['County']
                .apply(lambda x: sorted(list(x)))
                .to_dict()
            )

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

    def make_prediction(self, state: str = None, county: str = None,
                        year: int = None, race: str = None, sex: str = None) -> float:
        """Generate a prediction for life expectancy given input features."""
        
        # Create the input row with all features, initialized to NaN
        row = {col: np.nan for col in self.ALL_FEATURES}
        
        # Populate the row with the user's input
        if 'Year' in row: row['Year'] = year
        if 'Sex' in row: row['Sex'] = sex
        if 'Race' in row: row['Race'] = race
        if 'State' in row: row['State'] = state
        if 'County' in row: row['County'] = county

        # Create a DataFrame and make the prediction
        prediction_input = pd.DataFrame([row])
        prediction = self.STACKING_MODEL.predict(prediction_input)
        
        # Return the single prediction value
        return prediction[0]
