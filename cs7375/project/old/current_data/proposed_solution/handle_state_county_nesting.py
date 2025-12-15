#Create composite key when encoding State and County

df_county['StateCounty'] = df_county['State'] + '|' + df_county['County']

#Then in your pipeline:
categorical_features = ['State', 'StateCounty', 'Race', 'Sex']
numeric_features = ['Year', 'LifeExpectancyStandardError', 'DeathRate']
