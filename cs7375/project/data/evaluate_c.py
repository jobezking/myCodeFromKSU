# evaluate.py

import pandas as pd
import numpy as np
import sqlite3
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # required
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class Life_Expectancy_Evaluator:
    def __init__(self, test_size: float = 0.2, random_state: int = 42, cv_folds: int = 5):
        print("Initializing Evaluator: Loading data...")

        # Preprocessors
        self.iter_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=random_state)
        self.simple_imputer = SimpleImputer(strategy='most_frequent')
        self.encoder = OneHotEncoder(handle_unknown='ignore')

        # Config
        self.db_file = "life_expectancy_model_data.db"
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # Data containers
        self.full_dataset = None
        self.X = None
        self.y_stack = None

        # Splits and models
        self.splits = {}
        self.models = {}
        self.STACKING_MODEL = None

        # Results
        self.results = {}

        try:
            self._load_model_data()
            self._build_models()
            self._create_train_test_splits()
            self._train_base_models()
            self._build_stacking_features()
            self._train_meta_model()
            print("Evaluator is trained and ready.")
        except Exception as e:
            print(f"Initialization error: {e}")
            raise

    def _load_model_data(self):
        conn = sqlite3.connect(self.db_file)
        datasets = {
            "state_county_model": pd.read_sql("SELECT * FROM state_county_model_data", conn),
            "_133_year_race_sex_model": pd.read_sql("SELECT * FROM _133_year_race_sex_model_data", conn),
            "state_sex_model": pd.read_sql("SELECT * FROM state_sex_model_data", conn),
            "nchs_year_race_sex_model": pd.read_sql("SELECT * FROM nchs_year_race_sex_model_data", conn),
            "ssa_year_sex_model": pd.read_sql("SELECT * FROM ssa_year_sex_model_data", conn),
        }
        conn.close()

        for df in datasets.values():
            df.dropna(subset=['LifeExpectancy'], inplace=True)

        self.full_dataset = pd.concat(datasets.values(), ignore_index=True, sort=False)
        self.full_dataset = self.full_dataset.dropna(subset=['LifeExpectancy'])

        self.datasets = {k: (v.drop('LifeExpectancy', axis=1), v['LifeExpectancy']) for k, v in datasets.items()}
        self.X = self.full_dataset.drop('LifeExpectancy', axis=1)
        self.y_stack = self.full_dataset['LifeExpectancy']

    def _build_models(self):
        def pipeline(cols):
            return Pipeline([('transformer',
                              ColumnTransformer([(c, Pipeline([('imputer', self.simple_imputer),
                                                               ('encoder', self.encoder)]), [c])
                                                 if c in ['State', 'County', 'Sex', 'Race']
                                                 else (c, Pipeline([('imputer', self.iter_imputer),
                                                                    ('scaler', StandardScaler())]), [c])
                                                 for c in cols],
                                                remainder='drop')),
                             ('model', LinearRegression())])

        self.models = {
            "state_county_model": pipeline(['State', 'County']),
            "ssa_year_sex_model": pipeline(['Year', 'Sex']),
            "state_sex_model": pipeline(['State', 'Sex']),
            "nchs_year_race_sex_model": pipeline(['Year', 'Race', 'Sex']),
            "_133_year_race_sex_model": pipeline(['Year', 'Race', 'Sex']),
        }

    def _create_train_test_splits(self):
        for name, (X, y) in self.datasets.items():
            self.splits[name] = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def _train_base_models(self):
        for name, model in self.models.items():
            X_train, X_test, y_train, y_test = self.splits[name]
            model.fit(X_train, y_train)

    def _build_stacking_features(self):
        X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
            self.X, self.y_stack, test_size=self.test_size, random_state=self.random_state
        )
        self.meta_split = (X_meta_train, X_meta_test, y_meta_train, y_meta_test)

        preds = {}
        for name, model in self.models.items():
            cols = model.named_steps['transformer'].transformers
            req_cols = [c for c, _, _ in cols]
            preds[name] = cross_val_predict(model, X_meta_train[req_cols], y_meta_train, cv=self.kf)

        self.X_stack_train = pd.DataFrame({f"{name}_pred": p for name, p in preds.items()})
        self.y_stack_train = y_meta_train

        self.X_stack_test = pd.DataFrame({
            f"{name}_pred": self.models[name].predict(self.meta_split[1][[c for c, _, _ in
                                                                         self.models[name].named_steps['transformer'].transformers]])
            for name in self.models
        })
        self.y_stack_test = y_meta_test

        self.STACKING_MODEL = Pipeline([('imputer', self.iter_imputer), ('model', LinearRegression())])

    def _train_meta_model(self):
        self.STACKING_MODEL.fit(self.X_stack_train, self.y_stack_train)

    @staticmethod
    def _adjusted_r2(r2, n, p):
        return 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan

    def evaluate_models(self):
        results = {}
        for name, model in self.models.items():
            X_train, X_test, y_train, y_test = self.splits[name]
            y_pred = model.predict(X_test)
            n, p = X_test.shape
            results[name] = {
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAE": mean_absolute_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
                "Adj_R2": self._adjusted_r2(r2_score(y_test, y_pred), n, p),
                "CV_RMSE": -np.mean(cross_val_score(model, pd.concat([X_train, X_test]),
                                                    pd.concat([y_train, y_test]),
                                                    cv=self.kf, scoring='neg_root_mean_squared_error'))
            }

        y_pred_meta = self.STACKING_MODEL.predict(self.X_stack_test)
        n, p = self.X_stack_test.shape
        ols_model = sm.OLS(self.y_stack_train, sm.add_constant(self.X_stack_train)).fit()

        results["meta_model"] = {
            "RMSE": np.sqrt(mean_squared_error(self.y_stack_test, y_pred_meta)),
            "MAE": mean_absolute_error(self.y_stack_test, y_pred_meta),
            "R2": r2_score(self.y_stack_test, y_pred_meta),
            "Adj_R2": self._adjusted_r2(r2_score(self.y_stack_test, y_pred_meta), n, p),
            "CV_RMSE": -np.mean(cross_val_score(self.STACKING_MODEL, self.X_stack_train, self.y_stack_train,
                                                cv=self.kf, scoring='neg_root_mean_squared_error')),
            "F_statistic": ols_model.fvalue,
            "F_pvalue": ols_model.f_pvalue,
            "coef_tstats": ols_model.tvalues.to_dict(),
            "coef_pvalues": ols_model.pvalues.to_dict()
        }
        self.results = results
        return results

    def visualize_results(self, save_dir="plots"):
        os.makedirs(save_dir, exist_ok=True)
        metrics = ["RMSE", "MAE", "R2", "Adj_R2", "CV_RMSE"]
        df_metrics = pd.DataFrame(self.results).T[metrics].reset_index().rename(columns={'index': 'Model'})
        sns.set(style='whitegrid')

        # Metrics plot
        plt.figure(figsize=(12, 6))
        df_melt = df_metrics.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Value')
        sns.barplot(data=df_melt, x='Model', y='Value', hue='Metric')
        plt.title('Model Performance Metrics')
