import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer

# --- 1. Generate Simulated Data ---
print("Generating simulated housing-like data...")

# Numeric regression data
X_num, y = make_regression(
    n_samples=1000,
    n_features=6,
    n_informative=5,
    noise=25,
    random_state=42
)

num_feature_names = ["square_feet", "num_rooms", "age_years",
                     "lot_size", "distance_city", "school_rating"]
df_num = pd.DataFrame(X_num, columns=num_feature_names)

# Add categorical features
np.random.seed(42)
df_cat = pd.DataFrame({
    "has_pool": np.random.choice(["yes", "no"], size=1000),
    "garage_type": np.random.choice(["attached", "detached", "none"], size=1000),
    "neighborhood": np.random.choice(["A", "B", "C"], size=1000)
})

# Combine numeric + categorical
df = pd.concat([df_num, df_cat], axis=1)
target = pd.Series(y, name="price")

# --- 2. Define Feature Groups ---
X1 = df[["square_feet", "num_rooms", "age_years"]]          # House Specs (numeric)
X2 = df[["lot_size", "distance_city", "school_rating"]]     # Location Data (numeric)
X3 = df[["has_pool", "garage_type", "neighborhood"]]        # Amenities (categorical)

# --- 3. Train/Test Split ---
(X1_train, X1_test,
 X2_train, X2_test,
 X3_train, X3_test,
 y_train, y_test) = train_test_split(
    X1, X2, X3, target, test_size=0.2, random_state=42
)

# --- 4. Build Pipelines with ColumnTransformer ---
def build_pipeline(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", IterativeImputer(random_state=42), numeric_features),
            ("cat", SimpleImputer(strategy="most_frequent"), categorical_features)
        ],
        remainder="drop"
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])
    return pipeline

# Pipelines for each feature group
model_1 = build_pipeline(["square_feet", "num_rooms", "age_years"], [])
model_2 = build_pipeline(["lot_size", "distance_city", "school_rating"], [])
model_3 = build_pipeline([], ["has_pool", "garage_type", "neighborhood"])

# Train base models
print("Training base models...")
model_1.fit(X1_train, y_train)
model_2.fit(X2_train, y_train)
model_3.fit(X3_train, y_train)

# --- 5. Generate Meta-Features (Stacking) ---
print("Generating out-of-fold meta-features...")

preds_1_train = cross_val_predict(model_1, X1_train, y_train, cv=5)
preds_2_train = cross_val_predict(model_2, X2_train, y_train, cv=5)
preds_3_train = cross_val_predict(model_3, X3_train, y_train, cv=5)

X_meta_train = pd.DataFrame({
    "m1": preds_1_train,
    "m2": preds_2_train,
    "m3": preds_3_train
})

# --- 6. Train Meta-Model (with imputation safety) ---
meta_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", LinearRegression())
])
meta_pipeline.fit(X_meta_train, y_train)

weights = meta_pipeline.named_steps["model"].coef_
print(f"Meta-Model Weights: M1={weights[0]:.2f}, M2={weights[1]:.2f}, M3={weights[2]:.2f}")

# --- 7. Evaluate on Test Set ---
print("\n--- Evaluating Performance ---")

preds_1_test = model_1.predict(X1_test)
preds_2_test = model_2.predict(X2_test)
preds_3_test = model_3.predict(X3_test)

X_meta_test = pd.DataFrame({
    "m1": preds_1_test,
    "m2": preds_2_test,
    "m3": preds_3_test
})

final_predictions = meta_pipeline.predict(X_meta_test)

mse_1 = mean_squared_error(y_test, preds_1_test)
mse_2 = mean_squared_error(y_test, preds_2_test)
mse_3 = mean_squared_error(y_test, preds_3_test)
mse_meta = mean_squared_error(y_test, final_predictions)

print(f"Model 1 (Specs):     {mse_1:,.2f}")
print(f"Model 2 (Location):  {mse_2:,.2f}")
print(f"Model 3 (Amenities): {mse_3:,.2f}")
print("---------------------------------")
print(f"Ensemble Meta-Model: {mse_meta:,.2f}")

# --- 8. Inference Example ---
print("\n--- 🚀 New Prediction Example ---")

new_raw_data = df.iloc[X1_test.index[0]]
true_price = y_test.iloc[0]

# Split into same feature sets
new_data_1 = pd.DataFrame([new_raw_data[["square_feet","num_rooms","age_years"]]])
new_data_2 = pd.DataFrame([new_raw_data[["lot_size","distance_city","school_rating"]]])
new_data_3 = pd.DataFrame([new_raw_data[["has_pool","garage_type","neighborhood"]]])

# Base predictions (with imputation handling nulls if present)
pred_1 = model_1.predict(new_data_1)[0]
pred_2 = model_2.predict(new_data_2)[0]
pred_3 = model_3.predict(new_data_3)[0]

print(f"Base Predictions: M1={pred_1:.2f}, M2={pred_2:.2f}, M3={pred_3:.2f}")

# Meta-features
new_meta = pd.DataFrame([[pred_1, pred_2, pred_3]], columns=["m1","m2","m3"])
final_pred = meta_pipeline.predict(new_meta)[0]

print("\nFinal Result")
print("---------------------------------")
print(f"Final Ensemble Prediction: ${final_pred:,.2f}")
print(f"Actual House Price:        ${true_price:,.2f}")
print("---------------------------------")
