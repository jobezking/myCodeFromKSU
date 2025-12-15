import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # required for IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# -----------------------------
# Load datasets
# -----------------------------
df_county = pd.read_csv("life_expectancy_by_county_data.csv")
df_nchs   = pd.read_csv("nchs_cleaned.csv")

# -----------------------------
# County-level model
# -----------------------------
X_county = df_county[['State', 'County', 'LifeExpectancyStandardError']]
y_county = df_county['LifeExpectancy']

county_preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['State', 'County']),
    ('num', Pipeline([
        ('imputer', IterativeImputer(random_state=42, max_iter=10)),
        ('scaler', StandardScaler())
    ]), ['LifeExpectancyStandardError'])
])

county_model = Pipeline([
    ('preprocessor', county_preprocessor),
    ('regressor', LinearRegression())
])

county_model.fit(X_county, y_county)

# -----------------------------
# NCHS model
# -----------------------------
X_nchs = df_nchs[['Year', 'Race', 'Sex', 'AgeAdjustedDeathRate']]
y_nchs = df_nchs['LifeExpectancy']

nchs_preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Race', 'Sex']),
    ('num', Pipeline([
        ('imputer', IterativeImputer(random_state=42, max_iter=10)),
        ('scaler', StandardScaler())
    ]), ['Year', 'AgeAdjustedDeathRate'])
])

nchs_model = Pipeline([
    ('preprocessor', nchs_preprocessor),
    ('regressor', LinearRegression())
])

nchs_model.fit(X_nchs, y_nchs)

# -----------------------------
# Driver function
# -----------------------------
def predict_life_expectancy(state, county, year, race, sex):
    """
    User provides only mandatory fields.
    Optional numeric features are left as None and imputed.
    """
    # County input
    county_input = pd.DataFrame([{
        'State': state,
        'County': county,
        'LifeExpectancyStandardError': None  # imputed
    }])
    
    # NCHS input
    nchs_input = pd.DataFrame([{
        'Year': year,
        'Race': race,
        'Sex': sex,
        'AgeAdjustedDeathRate': None  # imputed
    }])
    
    # Predictions
    p1 = county_model.predict(county_input)[0]
    p2 = nchs_model.predict(nchs_input)[0]
    
    # Averaged prediction
    final_prediction = (p1 + p2) / 2
    return p1, p2, final_prediction

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    state = "Alabama"
    county = "Autauga County"
    year = 2010
    race = "All Races"
    sex = "Both Sexes"
    
    p1, p2, final = predict_life_expectancy(state, county, year, race, sex)
    
    print(f"County model prediction: {p1:.2f}")
    print(f"NCHS model prediction:   {p2:.2f}")
    print(f"Averaged prediction:     {final:.2f}")

