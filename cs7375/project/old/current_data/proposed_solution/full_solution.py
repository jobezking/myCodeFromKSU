import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Example: combined dataset (county + NCHS)
# df_all should have columns:
# ['State','County','Year','Race','Sex',
#  'LifeExpectancyStandardError','DeathRate','LifeExpectancy']

# Define features and target
X = df_all[['State', 'County', 'Year', 'Race', 'Sex',
            'LifeExpectancyStandardError', 'DeathRate']]
y = df_all['LifeExpectancy']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Mandatory features (must always be provided)
mandatory_categorical = ['State', 'County', 'Race', 'Sex']
mandatory_numeric = ['Year']

# Optional features (can be missing → imputed)
optional_numeric = ['LifeExpectancyStandardError', 'DeathRate']
optional_categorical = []  # add later if needed

# Transformers
mandatory_cat_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

mandatory_num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

optional_cat_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

optional_num_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer(random_state=42, max_iter=10)),
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
    'DeathRate': None                     # will be imputed
}])

prediction = model.predict(example)
print("Predicted Life Expectancy:", prediction[0])
