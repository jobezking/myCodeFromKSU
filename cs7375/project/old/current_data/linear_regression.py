import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

df = pd.read_csv('life_expectancy_by_county_data.csv',encoding='latin-1')
X = df[['State', 'County', 'LifeExpectancyStandardError']]
y = df['LifeExpectancy']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: one-hot encode categorical, scale numeric
categorical_features = ['State', 'County']
numeric_features = ['LifeExpectancyStandardError']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

# Build pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train
model.fit(X_train, y_train)

# Predict for new input
example = pd.DataFrame([{
    'State': 'Alabama',
    'County': 'Autauga County',
    'LifeExpectancyStandardError': 2.0
}])

prediction = model.predict(example)
print("Predicted Life Expectancy:", prediction[0])
