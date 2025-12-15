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
    def __init__(self, test_size=0.2, random_state=42, cv_folds=5):
        self.db_file = "life_expectancy_model_data.db"
        self.test_size = test_size
        self.random_state = random_state
        self.kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # Preprocessors
        self.iter_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=random_state)
        self.simple_imputer = SimpleImputer(strategy='most_frequent')
        self.encoder = OneHotEncoder(handle_unknown='ignore')

        # Containers
        self.datasets = {}
        self.models = {}
        self.splits = {}
        self.results = {}
        self.STACKING_MODEL = None

        print("Initializing Evaluator: Loading data...")
        self._load_model_data()
        self._build_models()
        self._create_train_test_splits()
        self._train_base_models()
        self._build_stacking_features()
        self._train_meta_model()
        print("Evaluator is trained and ready.")

    def _load_model_data(self):
        conn = sqlite3.connect(self.db_file)
        tables = {
            "state_county_model": "state_county_model_data",
            "_133_year_race_sex_model": "_133_year_race_sex_model_data",
            "state_sex_model": "state_sex_model_data",
            "nchs_year_race_sex_model": "nchs_year_race_sex_model_data",
            "ssa_year_sex_model": "ssa_year_sex_model_data",
        }
        for name, table in tables.items():
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            df.dropna(subset=['LifeExpectancy'], inplace=True)
            self.datasets[name] = (df.drop('LifeExpectancy', axis=1), df['LifeExpectancy'])
        conn.close()

        # Full dataset for meta-model
        self.full_dataset = pd.concat([pd.concat([X, y], axis=1) for X, y in self.datasets.values()],
                                      ignore_index=True, sort=False)
        self.full_dataset.dropna(subset=['LifeExpectancy'], inplace=True)
        self.X = self.full_dataset.drop('LifeExpectancy', axis=1)
        self.y = self.full_dataset['LifeExpectancy']

    def _build_models(self):
        def pipeline(cols):
            transformers = []
            for c in cols:
                if c in ['State', 'County', 'Sex', 'Race']:
                    transformers.append((c, Pipeline([('imputer', self.simple_imputer),
                                                      ('encoder', self.encoder)]), [c]))
                else:
                    transformers.append((c, Pipeline([('imputer', self.iter_imputer),
                                                      ('scaler', StandardScaler())]), [c]))
            return Pipeline([('transformer', ColumnTransformer(transformers, remainder='drop')),
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
            self.splits[name] = train_test_split(X, y, test_size=self.test_size,
                                                 random_state=self.random_state)

    def _train_base_models(self):
        for name, model in self.models.items():
            X_train, X_test, y_train, y_test = self.splits[name]
            model.fit(X_train, y_train)

    def _build_stacking_features(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)
        self.meta_split = (X_train, X_test, y_train, y_test)

        preds = {}
        for name, model in self.models.items():
            cols = [c for c, _, _ in model.named_steps['transformer'].transformers]
            preds[name] = cross_val_predict(model, X_train[cols], y_train, cv=self.kf)

        self.X_stack_train = pd.DataFrame({f"{name}_pred": p for name, p in preds.items()})
        self.y_stack_train = y_train.reset_index(drop=True)

        self.X_stack_test = pd.DataFrame({
            f"{name}_pred": self.models[name].predict(
                self.meta_split[1][[c for c, _, _ in self.models[name].named_steps['transformer'].transformers]]
            )
            for name in self.models
        })
        self.y_stack_test = y_test.reset_index(drop=True)

        self.STACKING_MODEL = Pipeline([('imputer', self.iter_imputer),
                                        ('model', LinearRegression())])

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
        ols_model = sm.OLS(self.y_stack_train.reset_index(drop=True),
                           sm.add_constant(self.X_stack_train.reset_index(drop=True))).fit()

        results["meta_model"] = {
            "RMSE": np.sqrt(mean_squared_error(self.y_stack_test, y_pred_meta)),
            "MAE": mean_absolute_error(self.y_stack_test, y_pred_meta),
            "R2": r2_score(self.y_stack_test, y_pred_meta),
            "Adj_R2": self._adjusted_r2(r2_score(self.y_stack_test, y_pred_meta), n, p),
            "CV_RMSE": -np.mean(cross_val_score(self.STACKING_MODEL, self.X_stack_train,
                                                self.y_stack_train, cv=self.kf,
                                                scoring='neg_root_mean_squared_error')),
            "F_statistic": ols_model.fvalue,
            "F_pvalue": ols_model.f_pvalue,
            "coef_tstats": ols_model.tvalues.to_dict(),
            "coef_pvalues": ols_model.pvalues.to_dict()
        }
        self.results = results
        return results

    def save_results(self, save_dir="plots"):
        os.makedirs(save_dir, exist_ok=True)
        metrics_file = os.path.join(save_dir, "metrics.txt")
        with open(metrics_file, "w") as f:
            f.write("=== Evaluation Results ===\n")
            for model, metrics in self.results.items():
                f.write(f"\nModel: {model}\n")
                for k, v in metrics.items():
                    # Handle nested dicts for meta-model coefficients
                    if isinstance(v, dict) and model == "meta_model" and k in ["coef_tstats", "coef_pvalues"]:
                        f.write(f"  {k}:\n")
                        # Build a table combining t-stats and p-values
                        if k == "coef_tstats":
                            coef_t = v
                            coef_p = metrics["coef_pvalues"]
                            f.write("    Coefficient        t-stat       p-value\n")
                            f.write("    ----------------------------------------\n")
                            for coef_name in coef_t.keys():
                                t_val = coef_t[coef_name]
                                p_val = coef_p.get(coef_name, np.nan)
                                f.write(f"    {coef_name:<15} {t_val:>10.4f} {p_val:>10.4f}\n")
                        # skip writing coef_pvalues separately (already included)
                    elif isinstance(v, dict):
                        f.write(f"  {k}:\n")
                        for subk, subv in v.items():
                            f.write(f"    {subk}: {subv}\n")
                    else:
                        f.write(f"  {k}: {v:.4f}\n")
        print(f"Metrics written to {metrics_file}")

    def visualize_results(self, save_dir="plots"):
        if not self.results:
            raise ValueError("No results found. Run evaluate() first.")

        os.makedirs(save_dir, exist_ok=True)

        # Metrics table
        metrics = ["RMSE", "MAE", "R2", "Adj_R2", "CV_RMSE"]
        df_metrics = pd.DataFrame(self.results).T[metrics].reset_index().rename(columns={"index": "Model"})
        sns.set(style="whitegrid")

        # Metrics plot
        plt.figure(figsize=(12, 6))
        df_melt = df_metrics.melt(id_vars="Model", value_vars=metrics,
                                  var_name="Metric", value_name="Value")
        sns.barplot(data=df_melt, x="Model", y="Value", hue="Metric")
        plt.title("Model Performance Metrics")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "metrics.png"))
        plt.close()

        # Meta-model stats
        meta_stats = self.results["meta_model"]

        # F-statistic
        plt.figure(figsize=(6, 4))
        sns.barplot(x=["F-statistic"], y=[meta_stats["F_statistic"]])
        plt.title("Meta-Model F-statistic")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "meta_f_statistic.png"))
        plt.close()

        # F p-value
        plt.figure(figsize=(6, 4))
        sns.barplot(x=["F p-value"], y=[meta_stats["F_pvalue"]])
        plt.title("Meta-Model F-statistic p-value")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "meta_f_pvalue.png"))
        plt.close()

        # Coefficient t-stats
        coef_t = pd.Series(meta_stats["coef_tstats"])
        plt.figure(figsize=(10, 6))
        sns.barplot(x=coef_t.index, y=coef_t.values)
        plt.title("Meta-Learner Coefficient t-statistics")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "meta_coef_tstats.png"))
        plt.close()

        # Coefficient p-values
        coef_p = pd.Series(meta_stats["coef_pvalues"])
        plt.figure(figsize=(10, 6))
        sns.barplot(x=coef_p.index, y=coef_p.values)
        plt.title("Meta-Learner Coefficient p-values")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "meta_coef_pvalues.png"))
        plt.close()

        print(f"Plots saved in {save_dir}/")


if __name__ == "__main__":
    evaluator = Life_Expectancy_Evaluator()
    results = evaluator.evaluate_models()
    evaluator.save_results(save_dir="plots")
    evaluator.visualize_results(save_dir="plots")
    print("Evaluation complete. See plots/metrics.txt and plots/*.png")
