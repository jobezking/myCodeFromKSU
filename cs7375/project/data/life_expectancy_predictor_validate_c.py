import pandas as pd
import sqlite3
import warnings
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Life_Expectancy_Predictor_Engine:
    def __init__(self, db_file="life_expectancy_model_data.db"):
        print("Initializing Engine: Loading data...")

        # imputers and encoders
        self.iter_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)
        self.simple_imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = OneHotEncoder(handle_unknown="ignore")

        # data holders
        self.db_file = db_file
        self.full_dataset = None
        self.ALL_FEATURES = None

        # train/test splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # metrics
        self.mse = {}
        self.rmse = {}
        self.r2score = {}

        # base models
        self.state_county_model = None
        self._133_year_race_sex_model = None
        self.state_sex_model = None
        self.nchs_year_race_sex_model = None
        self.ssa_year_sex_model = None

        # stacking
        self.X_stack_train = None
        self.STACKING_MODEL = None

        try:
            self._load_model_data()
            self._train_test_split()
            self._build_models_from_pipelines()
            self._train_models()
            self._build_stacking_model()
            print("Training the Stacking Ensemble Model...")
            self._train_stacking_model()
            print("Engine is trained and validated.")
        except Exception as e:
            print(f"Error during initialization: {e}")
            raise

    def _load_model_data(self):
        conn = sqlite3.connect(self.db_file)
        datasets = {
            "state_county": pd.read_sql("SELECT * FROM state_county_model_data", conn),
            "133": pd.read_sql("SELECT * FROM _133_year_race_sex_model_data", conn),
            "state_sex": pd.read_sql("SELECT * FROM state_sex_model_data", conn),
            "nchs": pd.read_sql("SELECT * FROM nchs_year_race_sex_model_data", conn),
            "ssa": pd.read_sql("SELECT * FROM ssa_year_sex_model_data", conn),
        }
        conn.close()

        for key in datasets:
            datasets[key].dropna(subset=["LifeExpectancy"], inplace=True)

        self.full_dataset = pd.concat(datasets.values(), ignore_index=True, sort=False)
        self.full_dataset.dropna(subset=["LifeExpectancy"], inplace=True)
        self.ALL_FEATURES = self.full_dataset.drop(columns=["LifeExpectancy"]).columns.tolist()

        self.datasets = datasets

    def _train_test_split(self):
        # Split the concatenated dataset for validation
        X = self.full_dataset.drop(columns=["LifeExpectancy"])
        y = self.full_dataset["LifeExpectancy"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def _build_pipeline(self, features):
        numeric_features = [f for f in features if f.lower() == "year"]
        categorical_features = [f for f in features if f not in numeric_features]

        transformers = []
        if numeric_features:
            transformers.append(
                ("num", Pipeline([("imputer", self.iter_imputer), ("scaler", StandardScaler())]), numeric_features)
            )
        if categorical_features:
            transformers.append(
                ("cat", Pipeline([("imputer", self.simple_imputer), ("encoder", self.encoder)]), categorical_features)
            )

        transformer = ColumnTransformer(transformers=transformers, remainder="drop")
        return Pipeline([("transformer", transformer), ("model", LinearRegression())])

    def _build_models_from_pipelines(self):
        self.state_county_model = self._build_pipeline(self.datasets["state_county"].drop(columns=["LifeExpectancy"]).columns)
        self._133_year_race_sex_model = self._build_pipeline(self.datasets["133"].drop(columns=["LifeExpectancy"]).columns)
        self.state_sex_model = self._build_pipeline(self.datasets["state_sex"].drop(columns=["LifeExpectancy"]).columns)
        self.nchs_year_race_sex_model = self._build_pipeline(self.datasets["nchs"].drop(columns=["LifeExpectancy"]).columns)
        self.ssa_year_sex_model = self._build_pipeline(self.datasets["ssa"].drop(columns=["LifeExpectancy"]).columns)

    def _train_models(self):
        # Fit each base model on its dataset
        self.state_county_model.fit(
            self.datasets["state_county"].drop(columns=["LifeExpectancy"]),
            self.datasets["state_county"]["LifeExpectancy"],
        )
        self._133_year_race_sex_model.fit(
            self.datasets["133"].drop(columns=["LifeExpectancy"]),
            self.datasets["133"]["LifeExpectancy"],
        )
        self.state_sex_model.fit(
            self.datasets["state_sex"].drop(columns=["LifeExpectancy"]),
            self.datasets["state_sex"]["LifeExpectancy"],
        )
        self.nchs_year_race_sex_model.fit(
            self.datasets["nchs"].drop(columns=["LifeExpectancy"]),
            self.datasets["nchs"]["LifeExpectancy"],
        )
        self.ssa_year_sex_model.fit(
            self.datasets["ssa"].drop(columns=["LifeExpectancy"]),
            self.datasets["ssa"]["LifeExpectancy"],
        )

    def _build_stacking_model(self):
        preds_state_county = cross_val_predict(self.state_county_model, self.X_train, self.y_train, cv=5)
        preds_133 = cross_val_predict(self._133_year_race_sex_model, self.X_train, self.y_train, cv=5)
        preds_state_sex = cross_val_predict(self.state_sex_model, self.X_train, self.y_train, cv=5)
        preds_nchs = cross_val_predict(self.nchs_year_race_sex_model, self.X_train, self.y_train, cv=5)
        preds_ssa = cross_val_predict(self.ssa_year_sex_model, self.X_train, self.y_train, cv=5)

        self.X_stack_train = pd.DataFrame(
            {
                "state_county_model_pred": preds_state_county,
                "_133_year_race_sex_model_pred": preds_133,
                "state_sex_model_pred": preds_state_sex,
                "nchs_year_race_sex_model_pred": preds_nchs,
                "ssa_year_sex_model_pred": preds_ssa,
            }
        )
        self.STACKING_MODEL = Pipeline([("imputer", IterativeImputer()), ("model", LinearRegression())])

    def _train_stacking_model(self):
        self.STACKING_MODEL.fit(self.X_stack_train, self.y_train)

        # Validation on test set
        preds_state_county = self.state_county_model.predict(self.X_test)
        preds_133 = self._133_year_race_sex_model.predict(self.X_test)
        preds_state_sex = self.state_sex_model.predict(self.X_test)
        preds_nchs = self.nchs_year_race_sex_model.predict(self.X_test)
        preds_ssa = self.ssa_year_sex_model.predict(self.X_test)

        X_stack_test = pd.DataFrame(
            {
                "state_county_model_pred": preds_state_county,
                "_133_year_race_sex_model_pred": preds_133,
                "state_sex_model_pred": preds_state_sex,
                "nchs_year_race_sex_model_pred": preds_nchs,
                "ssa_year_sex_model_pred": preds_ssa,
            }
        )
        final_preds = self.STACKING_MODEL.predict(X_stack_test)

        # compute metrics
        models = {
            "state_county": preds_state_county,
            "133": preds_133,
            "state_sex": preds_state_sex,
            "nchs": preds_nchs,
            "ssa": preds_ssa,
            "stacking": final_preds,
        }
        for name, preds in models.items():
            self.mse[name] = mean_squared_error(self.y_test, preds)
            self.rmse[name] = np.sqrt(self.mse[name])
            self.r2score[name] = r2_score(self.y_test, preds)

        self._plot_validation_results()

    def _plot_validation_results(self):
        metrics_df = pd.DataFrame(
            {"MSE": self.mse, "RMSE": self.rmse, "R2": self.r2score}
        ).T
        print("\nValidation Metrics:\n", metrics_df)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=metrics_df.index, y=metrics_df["RMSE"])
