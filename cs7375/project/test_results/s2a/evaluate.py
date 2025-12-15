import pandas as pd
import numpy as np
import sqlite3
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.inspection import permutation_importance

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
        
        # --- Imputers and Encoders ---
        self.iter_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)
        self.simple_imputer = SimpleImputer(strategy='most_frequent')
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 

        self.db_file = "life_expectancy_model_data.db"
        self.full_dataset = None
        
        # Model placeholders
        self.models = {} 
        self.STACKING_MODEL = None
        
        # Feature definitions mapping model names to specific columns
        self.feature_map = {
            'state_county_model': ['State', 'County'],
            'ssa_year_sex_model': ['Year', 'Sex'],
            'state_sex_model': ['State', 'Sex'],
            'nchs_year_race_sex_model': ['Year', 'Race', 'Sex'],
            '_133_year_race_sex_model': ['Year', 'Race', 'Sex']
        }

        # Metrics storage
        self.evaluation_results = []
        # Changed from self.meta_stats to self.meta_importance
        self.meta_importance = {} 

        try:
            self._load_model_data()
            self._build_pipelines()
            print("Data loaded and pipelines built. Ready for evaluation.")

        except sqlite3.OperationalError as e:
            print(f"CRITICAL ERROR: Database error. {e}")
            raise
        except Exception as e:
            print(f"An error occurred during initialization: {e}")
            raise

    def _load_model_data(self):
        datasets = []
        conn = sqlite3.connect(self.db_file)
        try:
            # Load specific tables present in the database
            datasets.append(pd.read_sql("SELECT * FROM state_county_model_data", conn))
            datasets.append(pd.read_sql("SELECT * FROM _133_year_race_sex_model_data", conn))
            datasets.append(pd.read_sql("SELECT * FROM state_sex_model_data", conn))
            datasets.append(pd.read_sql("SELECT * FROM nchs_year_race_sex_model_data", conn))
            datasets.append(pd.read_sql("SELECT * FROM ssa_year_sex_model_data", conn))
        finally:
            conn.close()

        # Create master dataset
        self.full_dataset = pd.concat(datasets, ignore_index=True, sort=False)
        self.full_dataset = self.full_dataset.dropna(subset=['LifeExpectancy'])
        self.full_dataset.reset_index(drop=True, inplace=True)
        
        self.X = self.full_dataset.drop('LifeExpectancy', axis=1)
        self.y = self.full_dataset['LifeExpectancy']

    def _build_pipelines(self):
        # Pipelines constructed based on original specifications
        
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

        # --- New Section 3: Calculate Permutation Importance ---
        # Permutation Importance is robust to multicollinearity and replaces the problematic OLS stats calculation.
        print("\n--- Calculating Permutation Importance for Meta-Model ---")
        
        r = permutation_importance(
            self.STACKING_MODEL, 
            X_meta_test, 
            y_test, 
            n_repeats=10, 
            random_state=42, 
            scoring='r2'
        )
        
        self.meta_importance = {
            'importances_mean': r.importances_mean,
            'importances_std': r.importances_std,
            'feature_names': meta_test_features.columns.tolist()
        }
        
        self.visualize_results()

    def visualize_results(self):
        """Creates visualizations and saves them to disk separately."""
        df_res = pd.DataFrame(self.evaluation_results)
        print("\nExecution Results:")
        print(df_res)
        
        # --- Visualizations ---
        sns.set_theme(style="whitegrid")
        
        # 1. Performance Metrics Comparison 
        metrics_to_plot = ['RMSE', 'MAE', 'R2', 'CV_RMSE']
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_res, x='Model', y=metric, palette='viridis')
            plt.xticks(rotation=45, ha='right')
            plt.title(f'{metric} by Model')
            plt.tight_layout()
            
            filename = f'model_performance_{metric}.png'
            plt.savefig(filename, dpi=300)
            print(f"Saved visualization: {filename}")
            plt.close()

        # 2. Meta-Model Permutation Importance (Replacing Significance Plot)
        if self.meta_importance:
            plt.figure(figsize=(12, 6))
            
            importance_df = pd.DataFrame({
                'Feature': self.meta_importance['feature_names'],
                'Importance': self.meta_importance['importances_mean'],
                'StdDev': self.meta_importance['importances_std']
            })
            
            # Filter out features with non-positive importance (i.e., base models that harm R2)
            importance_df = importance_df[importance_df['Importance'] > 0]
            importance_df = importance_df.sort_values(by='Importance', ascending=True)

            # Check if any features remain after filtering
            if not importance_df.empty:
                sns.barplot(data=importance_df, x='Importance', y='Feature', color='skyblue', xerr=importance_df['StdDev'])
                
                plt.title('Meta-Model Permutation Importance (Mean R2 Drop)')
                plt.xlabel('Importance (Mean Drop in R2 after Permutation)')
                plt.ylabel('Base Model Prediction')
                
                plt.tight_layout()
                filename_2 = 'meta_model_permutation_importance.png'
                plt.savefig(filename_2, dpi=300)
                print(f"Saved visualization: {filename_2}")
            else:
                print("Permutation Importance plot skipped: All feature importances were non-positive.")
            
            plt.close()

if __name__ == "__main__":
    engine = Life_Expectancy_Predictor_Engine()
    engine.run_evaluation()
