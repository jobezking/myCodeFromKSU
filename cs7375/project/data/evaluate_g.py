import pandas as pd
import numpy as np
import sqlite3
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class Life_Expectancy_Predictor_Engine:
    def __init__(self):
        print("Initializing Engine: Loading data...")
        
        # [cite_start]--- Imputers and Encoders [cite: 6] ---
        self.iter_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)
        self.simple_imputer = SimpleImputer(strategy='most_frequent')
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 

        self.db_file = "life_expectancy_model_data.db"
        self.full_dataset = None
        
        # Model placeholders
        self.models = {} 
        self.STACKING_MODEL = None
        
        # [cite_start]Feature definitions mapping model names to specific columns [cite: 8-10]
        self.feature_map = {
            'state_county_model': ['State', 'County'],
            'ssa_year_sex_model': ['Year', 'Sex'],
            'state_sex_model': ['State', 'Sex'],
            'nchs_year_race_sex_model': ['Year', 'Race', 'Sex'],
            '_133_year_race_sex_model': ['Year', 'Race', 'Sex']
        }

        # Metrics storage
        self.evaluation_results = []
        self.meta_stats = {}

        try:
            self._load_model_data()
            self._build_pipelines()
            print("Data loaded and pipelines built. Ready for evaluation.")

        except sqlite3.OperationalError:
            print(f"CRITICAL ERROR: Database file '{self.db_file}' not found.")
            raise
        except Exception as e:
            print(f"An error occurred during initialization: {e}")
            raise

    def _load_model_data(self):
        # [cite_start]Loading logic [cite: 19-21]
        datasets = []
        conn = sqlite3.connect(self.db_file)
        try:
            datasets.append(pd.read_sql("SELECT * FROM state_county_model_data", conn))
            datasets.append(pd.read_sql("SELECT * FROM _133_year_race_sex_model_data", conn))
            datasets.append(pd.read_sql("SELECT * FROM state_sex_model_data", conn))
            datasets.append(pd.read_sql("SELECT * FROM nchs_year_race_sex_model_data", conn))
            datasets.append(pd.read_sql("SELECT * FROM ssa_year_sex_model_data", conn))
        finally:
            conn.close()

        # [cite_start]Create master dataset [cite: 22]
        self.full_dataset = pd.concat(datasets, ignore_index=True, sort=False)
        self.full_dataset = self.full_dataset.dropna(subset=['LifeExpectancy'])
        self.full_dataset.reset_index(drop=True, inplace=True)
        
        self.X = self.full_dataset.drop('LifeExpectancy', axis=1)
        self.y = self.full_dataset['LifeExpectancy']

    def _build_pipelines(self):
        # [cite_start]Pipelines constructed based on original specifications [cite: 32-44]
        
        # SSA Year Sex
        t_ssa = ColumnTransformer([
            ('year', Pipeline([('imputer', self.iter_imputer), ('scaler', StandardScaler())]), ['Year']),
            ('sex', Pipeline([('imputer', self.simple_imputer), ('encoder', self.encoder)]), ['Sex'])
        ], remainder='drop')
        self.models['ssa_year_sex_model'] = Pipeline([('transformer', t_ssa), ('model', LinearRegression())])

        # State Sex
        t_ss = ColumnTransformer([
            ('state', Pipeline([('imputer', self.simple_imputer), ('encoder', self.encoder)]), ['State']),
            ('sex', Pipeline([('imputer', self.simple_imputer), ('encoder', self.encoder)]), ['Sex'])
        ], remainder='drop')
        self.models['state_sex_model'] = Pipeline([('transformer', t_ss), ('model', LinearRegression())])

        # State County
        t_sc = ColumnTransformer([
            ('state', Pipeline([('imputer', self.simple_imputer), ('encoder', self.encoder)]), ['State']),
            ('county', Pipeline([('imputer', self.simple_imputer), ('encoder', self.encoder)]), ['County'])
        ], remainder='drop')
        self.models['state_county_model'] = Pipeline([('transformer', t_sc), ('model', LinearRegression())])

        # NCHS
        t_nchs = ColumnTransformer([
            ('year', Pipeline([('imputer', self.iter_imputer), ('scaler', StandardScaler())]), ['Year']),
            ('sex', Pipeline([('imputer', self.simple_imputer), ('encoder', self.encoder)]), ['Sex']),
            ('race', Pipeline([('imputer', self.simple_imputer), ('encoder', self.encoder)]), ['Race'])
        ], remainder='drop')
        self.models['nchs_year_race_sex_model'] = Pipeline([('transformer', t_nchs), ('model', LinearRegression())])

        # 133
        t_133 = ColumnTransformer([
            ('year', Pipeline([('imputer', self.iter_imputer), ('scaler', StandardScaler())]), ['Year']),
            ('sex', Pipeline([('imputer', self.simple_imputer), ('encoder', self.encoder)]), ['Sex']),
            ('race', Pipeline([('imputer', self.simple_imputer), ('encoder', self.encoder)]), ['Race'])
        ], remainder='drop')
        self.models['_133_year_race_sex_model'] = Pipeline([('transformer', t_133), ('model', LinearRegression())])

    def calculate_adj_r2(self, r2, n, p):
        if n - p - 1 == 0: return 0
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def calculate_regression_stats(self, X, y, model, feature_names):
        """Calculates T-stats, P-values, and F-stat for Linear Regression."""
        predictions = model.predict(X)
        residuals = y - predictions
        n = len(y)
        p = X.shape[1]
        
        # Variance and Covariance
        sigma_squared = np.sum(residuals**2) / (n - p - 1)
        X_with_intercept = np.column_stack([np.ones(n), X])
        
        try:
            cov_matrix = sigma_squared * np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept))
            standard_errors = np.sqrt(np.diag(cov_matrix))
            
            # Coefficients
            params = np.insert(model.coef_, 0, model.intercept_)
            
            # T-stats and P-values
            t_stats = params / standard_errors
            p_values = [2 * (1 - stats.t.cdf(np.abs(t), n - p - 1)) for t in t_stats]
            
            # F-Statistic
            mean_y = np.mean(y)
            ss_total = np.sum((y - mean_y)**2)
            ss_res = np.sum(residuals**2)
            ss_reg = ss_total - ss_res
            f_stat = (ss_reg / p) / (ss_res / (n - p - 1))
            f_p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
            
            return {
                't_stats': t_stats, 
                'p_values': p_values, 
                'f_stat': f_stat, 
                'f_p_value': f_p_value,
                'coeffs': params,
                'names': ['Intercept'] + feature_names
            }
        except np.linalg.LinAlgError:
            print("Singular matrix encountered, cannot calculate advanced stats.")
            return None

    def run_evaluation(self):
        # 1. Split data into training and test sets
        print("Splitting data into Train (80%) and Test (20%)...")
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        meta_train_features = pd.DataFrame(index=X_train_full.index)
        meta_test_features = pd.DataFrame(index=X_test_full.index)

        # 2. Evaluate Base Models
        print("\n--- Evaluating Base Models ---")
        for name, pipeline in self.models.items():
            print(f"Processing {name}...")
            cols = self.feature_map[name]
            X_train_sub = X_train_full[cols]
            X_test_sub = X_test_full[cols]

            pipeline.fit(X_train_sub, y_train)

            # CV RMSE
            cv_scores = cross_val_score(pipeline, X_train_sub, y_train, 
                                        scoring='neg_root_mean_squared_error', cv=5)
            cv_rmse = -cv_scores.mean()

            # OOF predictions for Meta-Learner
            oof_preds = cross_val_predict(pipeline, X_train_sub, y_train, cv=5)
            meta_train_features[f"{name}_pred"] = oof_preds

            # Test Predictions
            y_pred = pipeline.predict(X_test_sub)
            meta_test_features[f"{name}_pred"] = y_pred

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            try:
                transformed_sample = pipeline.named_steps['transformer'].transform(X_test_sub.iloc[:5])
                p = transformed_sample.shape[1]
            except:
                p = len(cols)

            adj_r2 = self.calculate_adj_r2(r2, len(y_test), p)

            self.evaluation_results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Adj_R2': adj_r2,
                'CV_RMSE': cv_rmse
            })

        # --- Evaluate Meta Model ---
        print("\n--- Evaluating Stacking Meta-Model ---")
        
        self.STACKING_MODEL = LinearRegression()
        
        imputer = SimpleImputer(strategy='mean')
        X_meta_train = imputer.fit_transform(meta_train_features)
        X_meta_test = imputer.transform(meta_test_features)
        
        self.STACKING_MODEL.fit(X_meta_train, y_train)
        
        cv_scores_meta = cross_val_score(self.STACKING_MODEL, X_meta_train, y_train, 
                                         scoring='neg_root_mean_squared_error', cv=5)
        cv_rmse_meta = -cv_scores_meta.mean()

        y_pred_stack = self.STACKING_MODEL.predict(X_meta_test)
        
        rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))
        mae_stack = mean_absolute_error(y_test, y_pred_stack)
        r2_stack = r2_score(y_test, y_pred_stack)
        adj_r2_stack = self.calculate_adj_r2(r2_stack, len(y_test), X_meta_test.shape[1])

        self.evaluation_results.append({
            'Model': 'STACKING_META_MODEL',
            'RMSE': rmse_stack,
            'MAE': mae_stack,
            'R2': r2_stack,
            'Adj_R2': adj_r2_stack,
            'CV_RMSE': cv_rmse_meta
        })

        # 3. Calculate Stats
        stats_results = self.calculate_regression_stats(
            X_meta_test, y_test, self.STACKING_MODEL, meta_test_features.columns.tolist()
        )
        self.meta_stats = stats_results

        self.visualize_results()

    def visualize_results(self):
        """Creates visualizations and saves them to disk."""
        df_res = pd.DataFrame(self.evaluation_results)
        print("\nExecution Results:")
        print(df_res)
        
        if self.meta_stats:
            print("\nMeta-Model Statistical Significance:")
            print(f"F-Statistic: {self.meta_stats['f_stat']:.4f} (p={self.meta_stats['f_p_value']:.4e})")

        # --- Visualizations ---
        sns.set_theme(style="whitegrid")
        
        # 1. Performance Metrics Comparison (Saving to file)
        metrics_to_plot = ['RMSE', 'MAE', 'R2', 'CV_RMSE']
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Evaluation', fontsize=16)
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i//2, i%2]
            sns.barplot(data=df_res, x='Model', y=metric, ax=ax, palette='viridis')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title(f'{metric} by Model')
        
        plt.tight_layout()
        filename_1 = 'model_performance_metrics.png'
        plt.savefig(filename_1, dpi=300)
        print(f"Saved visualization: {filename_1}")
        plt.show()

        # 2. Meta-Model Statistical Significance (Saving to file)
        if self.meta_stats:
            plt.figure(figsize=(12, 6))
            
            stats_plot_df = pd.DataFrame({
                'Base Model (Weight)': self.meta_stats['names'],
                'T-Statistic': self.meta_stats['t_stats'],
                'P-Value': self.meta_stats['p_values']
            })
            
            stats_plot_df['Significance'] = stats_plot_df['P-Value'].apply(
                lambda x: 'Significant (p<0.05)' if x < 0.05 else 'Not Significant'
            )
            
            sns.barplot(data=stats_plot_df, x='Base Model (Weight)', y='T-Statistic', 
                        hue='Significance', dodge=False, palette='coolwarm')
            
            plt.title(f"Meta-Model Weights Significance\nModel F-Stat: {self.meta_stats['f_stat']:.2f} (p={self.meta_stats['f_p_value']:.2e})")
            plt.xticks(rotation=45, ha='right')
            plt.axhline(0, color='black', linewidth=1)
            
            plt.tight_layout()
            filename_2 = 'meta_model_coefficients_significance.png'
            plt.savefig(filename_2, dpi=300)
            print(f"Saved visualization: {filename_2}")
            plt.show()

if __name__ == "__main__":
    engine = Life_Expectancy_Predictor_Engine()
    engine.run_evaluation()