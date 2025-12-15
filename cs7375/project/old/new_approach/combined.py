import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

# Load input_a
a = pd.read_csv("input_a.csv")
a = a.drop(columns=["CensusTract"])
a["County"] = a["County"].str.replace(r",..$", "", regex=True)
a[["LifeExpectancyLow", "LifeExpectancyHigh"]] = (
    a["LifeExpectancyRange"].str.split("-", expand=True).astype(float)
)
a = a.groupby(["State", "County"], as_index=False).mean()

# Load input_b
b = pd.read_csv("input_b.csv")
b = b[["Year", "Race", "Sex", "Average Life Expectancy (Years)", "Age-adjusted Death Rate"]]

# Merge on Year only (broadcast national stats to all counties)
df = pd.merge(a, b, on="Year", how="left")

# Impute missing numeric values
imputer = SimpleImputer(strategy="mean")
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

print(df.head())
