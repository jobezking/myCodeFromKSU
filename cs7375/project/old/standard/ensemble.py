import pandas as pd
import sqlite3
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # required for IterativeImputer import
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import StackingRegressor


class LifeExpectancyPredictorEngine:
    def __init__(self, db_file: str = "life_expectancy_tables.db"):
        # imputers and encoders
        self.iter_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)
        self.simple_imputer = SimpleImputer(strategy='most_frequent')
        self.encoder = OneHotEncoder(handle_unknown='ignore')

        # database file
        self.db_file = db_file

        # placeholders
        self.full_dataset = None
        self.ALL_FEATURES = None
        self.STACKING_MODEL = None

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
        self._train_stacking_model()

    def _load_data(self):
        """Load tables from SQLite database into DataFrames."""
        conn = sqlite3.connect(self.db_file)
        sex_year = pd.read_sql("SELECT * FROM ssa_life_expectancy_sex_year", conn)
        sex_state = pd.read_sql("SELECT * FROM life_expectancy_sex_state", conn)
        state_county = pd.read_sql("SELECT * FROM life_expectancy_state_county", conn)
        race_sex = pd.read_sql("SELECT * FROM life_expectancy_race_sex", conn)
        race_sex_year = pd.read_sql("SELECT * FROM life_expectancy_race_sex_year", conn)
        conn.close()

        self.full_dataset = pd.concat([sex_year, sex_state, state_county, race_sex_year],
                                      ignore_index=True, sort=False)

    def _prepare_dataset(self):
        """Clean dataset and drop rows without target."""
        self.full_dataset = self.full_dataset.dropna(subset=['LifeExpectancy'])

    def _train_test_split(self):
        """Split dataset into train/test sets."""
        X = self.full_dataset.drop('LifeExpectancy', axis=1)
        y = self.full_dataset['LifeExpectancy']
        self.ALL_FEATURES = X.columns.tolist()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

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
            ]), ['Sex']),
            ('stderr', Pipeline([
                ('imputer', self.iter_imputer),
                ('scaler', StandardScaler())
            ]), ['StandardError']),
            ('le_quart_l', Pipeline([
                ('imputer', self.iter_imputer),
                ('scaler', StandardScaler())
            ]), ['LifeExpectancyQuartileLow']),
            ('le_quart_h', Pipeline([
                ('imputer', self.iter_imputer),
                ('scaler', StandardScaler())
            ]), ['LifeExpectancyQuartileHigh'])
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
            ]), ['County']),
            ('census', Pipeline([
                ('imputer', self.iter_imputer),
                ('scaler', StandardScaler())
            ]), ['CensusTract']),
            ('lstderr', Pipeline([
                ('imputer', self.iter_imputer),
                ('scaler', StandardScaler())
            ]), ['LifeExpectancyStandardError']),
            ('le_low', Pipeline([
                ('imputer', self.iter_imputer),
                ('scaler', StandardScaler())
            ]), ['LifeExpectancyLow']),
            ('le_high', Pipeline([
                ('imputer', self.iter_imputer),
                ('scaler', StandardScaler())
            ]), ['LifeExpectancyHigh'])
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

    def _build_stacking_model(self):
        estimators = [
            ('sex_year_model', self.sex_year_pipeline),
            ('sex_state_model', self.sex_state_pipeline),
            ('state_county_model', self.state_county_pipeline),
            ('race_sex_model', self.race_sex_pipeline)
        ]
        self.STACKING_MODEL = StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),
            cv=5,
            passthrough=False
        )

    def _train_stacking_model(self):
        self.STACKING_MODEL.fit(self.X_train, self.y_train)

    def make_prediction(self, state: str = None, county: str = None,
                        year: int = None, race: str = None, sex: str = None) -> float:
        """Generate a prediction for life expectancy given input features."""
        row = {col: np.nan for col in self.ALL_FEATURES}
        if 'Year' in row: row['Year'] = year
        if 'Sex' in row: row['Sex'] = sex
        if 'Race' in row: row['Race'] = race
        if 'State' in row: row['State'] = state
        if 'County' in row: row['County'] = county

        prediction_input = pd.DataFrame([row])
        prediction = self.STACKING_MODEL.predict(prediction_input)
        return prediction[0]


def predict_life_expectancy(state: str, county: str, year: int, race: str, sex: str) -> float:

    predictor = LifeExpectancyPredictorEngine()
    row = {col: np.nan for col in predictor.ALL_FEATURES}
    if 'Year' in row: row['Year'] = year
    if 'Sex' in row: row['Sex'] = sex
    if 'Race' in row: row['Race'] = race   
    if 'State' in row: row['State'] = state
    if 'County' in row: row['County'] = county

    prediction = predictor.STACKING_MODEL.predict(pd.DataFrame([row]))
    return prediction[0]
