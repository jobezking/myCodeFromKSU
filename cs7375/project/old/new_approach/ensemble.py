import pandas as pd
import numpy as np
import sqlite3
#
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # required for IterativeImputer import below
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import KFold  # for robust out-of-fold (OOF) base predictions used by the meta-model

# Note to future readers:
# This program implements an ensemble learning approach (stacking) with a meta-model that learns to
# weight three base models: geo_model (county-level geography), demo_model (demography), and geo_model_full
# (tract-level geography with imputation, used in a tract-agnostic manner at inference).
# We retain the original variable names and overall program flow, but replace the final averaging step
# with a learned meta-model. The meta-model is trained on out-of-fold predictions from the base models
# to reduce overfitting and to learn optimal weights. At inference, we compute base predictions and feed
# them into the meta-model for the final life expectancy estimate.
#
# Key design choices:
# - All imputations remain inside ColumnTransformer pipelines, so passing None at inference is supported.
# - We avoid merging tract-level df_geo_full with county-level df_geo for training the base learners.
# - For meta-model training, we merge df_demo with df_geo using the common key "State_County" to gather
#   overlapping samples. This overlap may be smaller than either dataset, which is fine; we mitigate via
#   regularization (BayesianRidge) and out-of-fold predictions.
# - At inference, df_geo_full base prediction is computed using a synthetic tract label "unknown" to allow
#   OneHotEncoder to handle unseen categories (handled via handle_unknown='ignore') and imputers to fill
#   numeric fields.


def life_expectancy_predictor_engine(geo_input: pd.DataFrame, demo_input: pd.DataFrame) -> float:
    db_file = "life_expectancy.db"
    db_file_geo_full = "life_expectancy_geo_full.db"
    table_geo = "life_expectancy_geography"
    table_demo = "life_expectancy_demography"
    table_geo_full = "life_expectancy_geography_full"

    # ---------------------------------------------------------------------------------------------
    # Load training data from SQLite
    # ---------------------------------------------------------------------------------------------
    conn1 = sqlite3.connect(db_file)
    conn2 = sqlite3.connect(db_file_geo_full)
    df_geo = pd.read_sql(f"SELECT * FROM {table_geo}", conn1)
    df_demo = pd.read_sql(f"SELECT * FROM {table_demo}", conn1)
    df_geo_full = pd.read_sql(f"SELECT * FROM {table_geo_full}", conn2)
    conn1.close()
    conn2.close()

    # ---------------------------------------------------------------------------------------------
    # Key feature engineering for IDs used in pipelines
    # ---------------------------------------------------------------------------------------------
    # County-level composite key used by geo_model
    df_geo['State_County'] = df_geo['State'] + '|' + df_geo['County']
    # Tract-level composite key used by geo_model_full
    df_geo_full['State_County_Census'] = (
        df_geo_full['State'] + '|' + df_geo_full['County'] + '|' + df_geo_full['CensusTract'].astype(str)
    )

    # ---------------------------------------------------------------------------------------------
    # Define feature sets and target columns per dataset
    # ---------------------------------------------------------------------------------------------
    X_geo = ['State_County', 'LifeExpectancyStandardError']
    y_geo = 'LifeExpectancy'

    X_demo = ['Year', 'Race', 'Sex', 'AgeAdjustedDeathRate']
    y_demo = 'LifeExpectancy'

    # For df_geo_full, impute its internal numeric columns prior to pipeline training. This ensures
    # IterativeImputer has a consistent basis and helps stabilize downstream model fitting.
    numeric_cols = ["LifeExpectancy", "LifeExpectancyLow", "LifeExpectancyHigh", "LifeExpectancyStandardError"]
    imputer_y = IterativeImputer(estimator=BayesianRidge(), random_state=42)
    imputed = imputer_y.fit_transform(df_geo_full[numeric_cols])
    df_geo_full[numeric_cols] = imputed

    X_geo_full = ['State_County_Census', 'LifeExpectancyStandardError', 'LifeExpectancyLow', 'LifeExpectancyHigh']
    y_geo_full = 'LifeExpectancy'

    # ---------------------------------------------------------------------------------------------
    # Define model-based imputers and preprocessors
    # ---------------------------------------------------------------------------------------------
    iter_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)

    # Preprocessor for tract-level geography (df_geo_full)
    geo_preprocessor_full = ColumnTransformer(transformers=[
        # Categorical pipeline for State|County|CensusTract key:
        ('statecountycensus', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # categorical imputations
            ('encoder', OneHotEncoder(handle_unknown='ignore'))    # robust to unseen categories
        ]), ['State_County_Census']),
        # Numeric pipelines with model-based imputation and scaling:
        ('lifeexpectlow', Pipeline([
            ('imputer', iter_imputer),
            ('scaler', StandardScaler())
        ]), ['LifeExpectancyLow']),
        ('lifeexpecthigh', Pipeline([
            ('imputer', iter_imputer),
            ('scaler', StandardScaler())
        ]), ['LifeExpectancyHigh']),
        ('stderr', Pipeline([
            ('imputer', iter_imputer),
            ('scaler', StandardScaler())
        ]), ['LifeExpectancyStandardError']),
    ])

    # Preprocessor for county-level geography (df_geo)
    geo_preprocessor = ColumnTransformer(transformers=[
        ('statecounty', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), ['State_County']),
        ('stderr', Pipeline([
            ('imputer', iter_imputer),
            ('scaler', StandardScaler())
        ]), ['LifeExpectancyStandardError'])
    ])

    # Preprocessor for demography (df_demo)
    demo_preprocessor = ColumnTransformer(transformers=[
        ('year', Pipeline([
            ('imputer', iter_imputer),
            ('scaler', StandardScaler())
        ]), ['Year']),
        ('race', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), ['Race']),
        ('sex', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), ['Sex']),
        ('deathrate', Pipeline([
            ('imputer', iter_imputer),
            ('scaler', StandardScaler())
        ]), ['AgeAdjustedDeathRate'])
    ])

    # ---------------------------------------------------------------------------------------------
    # Build base models (unchanged variable names)
    # ---------------------------------------------------------------------------------------------
    geo_model = Pipeline(steps=[
        ('preprocessor', geo_preprocessor),
        ('regressor', LinearRegression())
    ])

    demo_model = Pipeline(steps=[
        ('preprocessor', demo_preprocessor),
        ('regressor', LinearRegression())
    ])

    geo_model_full = Pipeline(steps=[
        ('preprocessor', geo_preprocessor_full),
        ('regressor', LinearRegression())
    ])

    # ---------------------------------------------------------------------------------------------
    # Train base models on their respective datasets
    # ---------------------------------------------------------------------------------------------
    # These fits use internal imputers and encoders, so NaNs and unseen categories at inference are handled.
    geo_model.fit(df_geo[X_geo], df_geo[y_geo])
    demo_model.fit(df_demo[X_demo], df_demo[y_demo])
    geo_model_full.fit(df_geo_full[X_geo_full], df_geo_full[y_geo_full])

    # ---------------------------------------------------------------------------------------------
    # Train meta-model (stacking) using overlapping county-level records
    # ---------------------------------------------------------------------------------------------
    # Strategy:
    # - Create a county composite key in df_demo to match df_geo.
    # - Inner-join on State_County to get overlapping rows (shared samples).
    # - For these shared samples, compute out-of-fold predictions from geo_model and demo_model.
    # - Also compute tract-agnostic geo_model_full predictions by synthesizing a State_County_Census value.
    # - Fit a regularized meta-model (BayesianRidge) on the stacked OOF predictions to learn optimal weights.

    # Construct county key in df_demo for merging
    if 'State_County' not in df_demo.columns:
        # Assumes df_demo also contains State and County; if not, adjust upstream ETL to carry them through.
        df_demo['State_County'] = df_demo['State'] + '|' + df_demo['County']

    # Inner join to get overlap; this will reduce sample size to common keys, which is expected
    df_overlap = df_demo.merge(
        df_geo[['State_County', 'LifeExpectancy', 'LifeExpectancyStandardError']],
        on='State_County',
        how='inner',
        suffixes=('_demo', '_geo')
    )

    # If the overlap is very small, consider fallbacks (e.g., two-model meta). Here, we proceed if we have rows.
    # Build base-model feature frames for the overlap
    X_geo_overlap = df_overlap[['State_County', 'LifeExpectancyStandardError']]
    X_demo_overlap = df_overlap[['Year', 'Race', 'Sex', 'AgeAdjustedDeathRate']]
    # Use county-level life expectancy (from df_geo) as target for meta-learner
    y_meta = df_overlap['LifeExpectancy']

    # Prepare tract-agnostic features for geo_model_full by synthesizing "unknown" tract.
    # OneHotEncoder(handle_unknown='ignore') ensures this category is safely ignored without error.
    df_overlap['State_County_Census'] = df_overlap['State_County'] + '|unknown'
    X_full_overlap = df_overlap[['State_County_Census', 'LifeExpectancyLow', 'LifeExpectancyHigh', 'LifeExpectancyStandardError']]

    # Out-of-fold prediction containers for each base model
    oof_geo = np.zeros(len(df_overlap))
    oof_demo = np.zeros(len(df_overlap))
    oof_full = np.zeros(len(df_overlap))

    # K-Fold setup for robust meta-features (OOF predictions avoid in-sample bias)
    # Choose n_splits based on overlap size; 5 is a typical default.
    if len(df_overlap) >= 5:
        n_splits = 5
    else:
        # If very few samples overlap, fallback to leave-one-out-like behavior (min 2 splits),
        # acknowledging higher variance. Adjust as needed.
        n_splits = max(2, len(df_overlap))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Generate OOF predictions
    for train_idx, valid_idx in kf.split(df_overlap):
        # Slice training/validation folds
        Xg_train = X_geo_overlap.iloc[train_idx]
        Xg_valid = X_geo_overlap.iloc[valid_idx]
        Xd_train = X_demo_overlap.iloc[train_idx]
        Xd_valid = X_demo_overlap.iloc[valid_idx]
        Xf_train = X_full_overlap.iloc[train_idx]
        Xf_valid = X_full_overlap.iloc[valid_idx]
        y_train = y_meta.iloc[train_idx]

        # Clone and fit base models within the fold.
        # Using the same pipelines to preserve preprocessing behavior.
        geo_fold = Pipeline(steps=geo_model.steps)
        demo_fold = Pipeline(steps=demo_model.steps)
        geo_full_fold = Pipeline(steps=geo_model_full.steps)

        geo_fold.fit(Xg_train, y_train)
        demo_fold.fit(Xd_train, y_train)
        geo_full_fold.fit(Xf_train, y_train)

        # Validation predictions (OOF)
        oof_geo[valid_idx] = geo_fold.predict(Xg_valid)
        oof_demo[valid_idx] = demo_fold.predict(Xd_valid)
        oof_full[valid_idx] = geo_full_fold.predict(Xf_valid)

    # Stack OOF base predictions to form meta-features
    Z_meta = np.column_stack([oof_geo, oof_demo, oof_full])

    # Train a regularized meta-learner to learn the weighting and interactions of base predictions
    meta_learner = BayesianRidge()
    meta_learner.fit(Z_meta, y_meta)

    # ---------------------------------------------------------------------------------------------
    # Inference: compute base predictions from user inputs and combine via the meta-model
    # ---------------------------------------------------------------------------------------------
    # User-driven application constraints:
    # - geo_input comes with only: State_County and LifeExpectancyStandardError (None imputable)
    # - demo_input comes with only: Year, Race, Sex, AgeAdjustedDeathRate (None imputable)
    # - geo_model_full requires tract and extra numeric features; we synthesize tract "unknown"
    #   and provide None for numeric fields so imputers can fill them.

    # Base predictions from county-level geo and demography models:
    geo_pred = geo_model.predict(geo_input)
    demo_pred = demo_model.predict(demo_input)

    # Construct tract-agnostic input for df_geo_full model
    # We follow your attached approach: build a geo_full_input with None values for numeric fields,
    # and create a State_County_Census using "unknown" as the tract.
    state_county_val = geo_input.iloc[0]['State_County']
    geo_full_input = pd.DataFrame([{
        'State_County_Census': f"{state_county_val}|unknown",   # synthetic tract label
        'LifeExpectancyStandardError': None,
        'LifeExpectancyLow': None,
        'LifeExpectancyHigh': None
    }])

    geo_full_pred = geo_model_full.predict(geo_full_input)

    # Compose meta-features for this single inference row.
    # Ensure the column order matches Z_meta training order: [geo, demo, full]
    Z_infer = np.column_stack([geo_pred, demo_pred, geo_full_pred])

    # Final stacked prediction from meta-model
    final_pred = meta_learner.predict(Z_infer)

    # Return the meta-model output; shape is array-like, but function expects float.
    # If batch predictions are later required, adjust return type accordingly.
    return float(final_pred[0])


def predict_life_expectancy(state: str, county: str, year: int, race: str, sex: str) -> float:
    """
    Receives inputs from Flask, builds input DataFrames,
    calls life_expectancy_predictor_engine, and returns the float estimate.

    Design notes:
    - Only user-provided fields are filled; other features are left as None so the pipelines' imputers
      can estimate them.
    - State_County is the composite categorical feature key used by geo_model.
    - AgeAdjustedDeathRate is None here and will be imputed within demo_model's pipeline.
    """

    state_county = f"{state}|{county}"  # composite key
    geo_input = pd.DataFrame([{
        'State_County': state_county,
        'LifeExpectancyStandardError': None  # None is acceptable; IterativeImputer will handle it
    }])
    demo_input = pd.DataFrame([{
        'Year': year,
        'Race': race,
        'Sex': sex,
        'AgeAdjustedDeathRate': None  # None imputable by IterativeImputer
    }])

    return life_expectancy_predictor_engine(geo_input, demo_input)
