import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Example: combining both datasets
# df_county = pd.read_csv("county_life_expectancy.csv")
# df_nchs = pd.read_csv("NCHS_-_Death_rates_and_life_expectancy_at_birth.csv")

# Harmonize column names
df_nchs = df_nchs.rename(columns={
    "Average Life Expectancy (Years)": "LifeExpectancy",
    "Age-adjusted Death Rate": "DeathRate"
})

# Add missing columns to align with county dataset
df_nchs["State"] = None
df_nchs["County"] = None
df_nchs["LifeExpectancyStandardError"] = None

# Concatenate datasets
df_all = pd.concat([df_county, df_nchs], ignore_index=True)

# Define features and target
X = df_all[['State', 'County', 'Year', 'Race', 'Sex',
            'LifeExpectancyStandardError', 'DeathRate']]
y = df_all['LifeExpectancy']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature groups
categorical_features = ['State', 'County', 'Race', 'Sex']
numeric_features = ['Year', 'LifeExpectancyStandardError', 'DeathRate']

# Pipelines
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features)
    ]
)

# Final pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train
model.fit(X_train, y_train)

# Example prediction
example = pd.DataFrame([{
    'State': 'Alabama',
    'County': 'Autauga County',
    'Year': 2010,
    'Race': 'All Races',
    'Sex': 'Both Sexes',
    'LifeExpectancyStandardError': None,  # will be imputed
    'DeathRate': 747.0
}])

prediction = model.predict(example)
print("Predicted Life Expectancy:", prediction[0])
