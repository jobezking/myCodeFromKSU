import pandas as pd

# Import scikit-learn tools
from sklearn.experimental import enable_iterative_imputer  # This line is REQUIRED to enable IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# ============================================================
# 1. Load datasets
# ============================================================
df_county = pd.read_csv("life_expectancy_by_county_data.csv")
df_nchs   = pd.read_csv("nchs_cleaned.csv")

# ============================================================
# 2. Preprocess County dataset
# ============================================================

# Create a composite key "StateCounty" so that counties with the same name
# in different states are treated as distinct categories.
df_county['StateCounty'] = df_county['State'] + '|' + df_county['County']

# Define features (X) and target (y) for the county model
X_county = df_county[['StateCounty', 'LifeExpectancyStandardError']]
y_county = df_county['LifeExpectancy']

# -----------------------------
# County categorical transformer
# -----------------------------
# OneHotEncoder will turn the "StateCounty" column into binary indicator columns.
# Example: "Alabama|Autauga County" becomes a column with 1 if present, 0 otherwise.
county_cat_encoder = OneHotEncoder(handle_unknown='ignore')

# -----------------------------
# County numeric transformer
# -----------------------------
# IterativeImputer is a model-based imputer.
# It works by treating each column with missing values as a regression target,
# and predicting it using the other columns.
#
# In this case, the only numeric feature is LifeExpectancyStandardError.
# If it's missing, IterativeImputer will try to estimate it using patterns
# in the dataset (though with only one numeric feature, it effectively reduces
# to something like mean imputation).
#
# StandardScaler then standardizes the numeric column (mean=0, std=1).
county_num_imputer = IterativeImputer(
    random_state=42,  # ensures reproducibility
    max_iter=10       # number of imputation iterations
)
county_num_scaler = StandardScaler()

# Build numeric pipeline step by step
from sklearn.pipeline import make_pipeline
county_num_pipeline = make_pipeline(county_num_imputer, county_num_scaler)

# -----------------------------
# County preprocessor
# -----------------------------
# ColumnTransformer applies different preprocessing to different columns.
county_preprocessor = ColumnTransformer([
    ('cat', county_cat_encoder, ['StateCounty']),
    ('num', county_num_pipeline, ['LifeExpectancyStandardError'])
])

# -----------------------------
# County model pipeline
# -----------------------------
county_model = Pipeline([
    ('preprocessor', county_preprocessor),
    ('regressor', LinearRegression())
])

# Train the county model
county_model.fit(X_county, y_county)

# ============================================================
# 3. Preprocess NCHS dataset
# ============================================================

# Define features (X) and target (y) for the NCHS model
X_nchs = df_nchs[['Year', 'Race', 'Sex', 'AgeAdjustedDeathRate']]
y_nchs = df_nchs['LifeExpectancy']

# -----------------------------
# NCHS categorical transformer
# -----------------------------
nchs_cat_encoder = OneHotEncoder(handle_unknown='ignore')

# -----------------------------
# NCHS numeric transformer
# -----------------------------
# Here we have two numeric features: Year and AgeAdjustedDeathRate.
# If either is missing, IterativeImputer will try to predict it
# using the other numeric feature(s).
nchs_num_imputer = IterativeImputer(
    random_state=42,
    max_iter=10
)
nchs_num_scaler = StandardScaler()
nchs_num_pipeline = make_pipeline(nchs_num_imputer, nchs_num_scaler)

# -----------------------------
# NCHS preprocessor
# -----------------------------
nchs_preprocessor = ColumnTransformer([
    ('cat', nchs_cat_encoder, ['Race', 'Sex']),
    ('num', nchs_num_pipeline, ['Year', 'AgeAdjustedDeathRate'])
])

# -----------------------------
# NCHS model pipeline
# -----------------------------
nchs_model = Pipeline([
    ('preprocessor', nchs_preprocessor),
    ('regressor', LinearRegression())
])

# Train the NCHS model
nchs_model.fit(X_nchs, y_nchs)

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
    are set to None and will be imputed automatically by IterativeImputer.
    """

    # Build composite key for county input
    state_county = f"{state}|{county}"

    # County input (LifeExpectancyStandardError left as None → imputed)
    county_input = pd.DataFrame([{
        'StateCounty': state_county,
        'LifeExpectancyStandardError': None
    }])

    # NCHS input (AgeAdjustedDeathRate left as None → imputed)
    nchs_input = pd.DataFrame([{
        'Year': year,
        'Race': race,
        'Sex': sex,
        'AgeAdjustedDeathRate': None
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
    state = "Mississippi"
    county = "Washington County"
    year = 2010
    race = "All Races"
    sex = "Both Sexes"

    p1, p2, final = predict_life_expectancy(state, county, year, race, sex)

    print(f"County model prediction: {p1:.2f}")
    print(f"NCHS model prediction:   {p2:.2f}")
    print(f"Averaged prediction:     {final:.2f}")

