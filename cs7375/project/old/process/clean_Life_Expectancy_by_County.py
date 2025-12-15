import pandas as pd

# Load the CSV file
# Adjust the path to where your file is located
df = pd.read_csv("U.S._Life_Expectancy_at_Birth_by_State_and_Census_Tract_-_2010-2015.csv")

# Inspect columns (uncomment if needed)
# print(df.columns)

# Drop unnecessary columns: census tract number and life expectancy range
# Adjust names if they differ slightly in your dataset
cols_to_drop = ["CensusTract", "LifeExpectancyRange"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Remove rows without county information
#df = df[df["County"].notna()]

# Strip whitespace and drop rows where County equals "(blank)"
df = df[df["County"].str.strip() != "(blank)"]


# Group by state and county, aggregating life expectancy and standard error
# Using mean as the approximation method
agg_df = (
    df.groupby(["State", "County"], as_index=False)
      .agg({
          "LifeExpectancy": "mean",
          "LifeExpectancyStandardError": "mean"
      })
)

# Optional: round values for readability
agg_df["LifeExpectancy"] = agg_df["LifeExpectancy"].round(2)
agg_df["LifeExpectancyStandardError"] = agg_df["LifeExpectancyStandardError"].round(2)

# Final DataFrame
print(agg_df.head(10))
agg_df.to_csv("U.S._Life_Expectancy_at_Birth_by_State_and_Census_Tract_2010-2015_cleaned.csv", index=False)
