import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ============================================================
# 1. Load datasets
# ============================================================
county_df = pd.read_csv("life_expectancy_by_county_data.csv")
nchs_df   = pd.read_csv("nchs_cleaned.csv")

# ============================================================
# 2. Preprocess County dataset
# ============================================================

# Create composite key "StateCounty" so that counties with the same name
# in different states are treated as distinct categories.
county_df['StateCounty'] = county_df['State'] + '|' + county_df['County']

# Define features (X) and target (y) for the county model
county_features = ['StateCounty', 'LifeExpectancyStandardError']
county_target   = 'LifeExpectancy'

# -----------------------------
# County preprocessing
# -----------------------------
# - StateCounty: categorical → impute most frequent → one-hot encode
# - LifeExpectancyStandardError: numeric → impute median → scale
county_preprocessor = ColumnTransformer(transformers=[
    ('statecounty', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), ['StateCounty']),
    ('stderr', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), ['LifeExpectancyStandardError'])
])

# Build county model pipeline
county_model = Pipeline(steps=[
    ('preprocessor', county_preprocessor),
    ('regressor', LinearRegression())
])

# Train county model
county_model.fit(county_df[county_features], county_df[county_target])

# ============================================================
# 3. Preprocess NCHS dataset
# ============================================================

nchs_features = ['Year', 'Race', 'Sex', 'AgeAdjustedDeathRate']
nchs_target   = 'LifeExpectancy'

# -----------------------------
# NCHS preprocessing
# -----------------------------
# - Year, AgeAdjustedDeathRate: numeric → impute median → scale
# - Race, Sex: categorical → impute most frequent → one-hot encode
nchs_preprocessor = ColumnTransformer(transformers=[
    ('year', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
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
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), ['AgeAdjustedDeathRate'])
])

# Build NCHS model pipeline
nchs_model = Pipeline(steps=[
    ('preprocessor', nchs_preprocessor),
    ('regressor', LinearRegression())
])

# Train NCHS model
nchs_model.fit(nchs_df[nchs_features], nchs_df[nchs_target])

# ============================================================
# 4. Driver function for predictions
# ============================================================

def predict_life_expectancy(state, county, year, race, sex):
    """
    User provides only the mandatory fields:
    - State
    - County
    - Year
    - Race
    - Sex

    Optional numeric features (LifeExpectancyStandardError, AgeAdjustedDeathRate)
    are set to NaN and imputed automatically by SimpleImputer.
    """

    # Build composite key for county input
    state_county = f"{state}|{county}"

    # County input (LifeExpectancyStandardError left as NaN → imputed)
    county_input = pd.DataFrame([{
        'StateCounty': state_county,
        'LifeExpectancyStandardError': np.nan
    }])

    # NCHS input (AgeAdjustedDeathRate left as NaN → imputed)
    nchs_input = pd.DataFrame([{
        'Year': year,
        'Race': race,
        'Sex': sex,
        'AgeAdjustedDeathRate': np.nan
    }])

    # Predictions from each model
    p1 = county_model.predict(county_input)[0]
    p2 = nchs_model.predict(nchs_input)[0]

    # Average the predictions
    final_prediction = (p1 + p2) / 2

    return p1, p2, final_prediction

# ============================================================
# 5. Example usage
# ============================================================
if __name__ == "__main__":
    # Example: user provides only the mandatory fields
    state = "Alabama"
    county = "Autauga County"
    year = 2010
    race = "All Races"
    sex = "Both Sexes"

    p1, p2, final = predict_life_expectancy(state, county, year, race, sex)

    print(f"County model prediction: {p1:.2f}")
    print(f"NCHS model prediction:   {p2:.2f}")
    print(f"Averaged prediction:     {final:.2f}")

