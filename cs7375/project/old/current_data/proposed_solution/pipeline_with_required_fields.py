from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Define mandatory and optional features
mandatory_categorical = ['State', 'County', 'Race', 'Sex']
mandatory_numeric = ['Year']

optional_numeric = ['LifeExpectancyStandardError', 'DeathRate']
optional_categorical = []  # add if you have more later

# Transformers
mandatory_cat_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

mandatory_num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

optional_cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

optional_num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('mand_cat', mandatory_cat_transformer, mandatory_categorical),
        ('mand_num', mandatory_num_transformer, mandatory_numeric),
        ('opt_cat', optional_cat_transformer, optional_categorical),
        ('opt_num', optional_num_transformer, optional_numeric)
    ]
)

# Final pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
