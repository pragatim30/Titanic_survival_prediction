import pandas as pd
import numpy as np

df = pd.read_csv('data/titanic_eda_features.csv')
print("Shape:", df.shape)
print("Target balance (survived):\n", df['survived'].value_counts(normalize=True).round(3))

# Build the final feature set for modeling.
# Reuse engineered features from the Week 2 EDA (family_size, has_cabin_record, etc.)
# plus one new feature for this task: is_child (age < 16), since EDA in Week 2 showed
# a sharp survival difference tied to age that pclass/sex alone don't fully capture.
df['is_child'] = (df['age'] < 16).astype(int)

feature_cols = [
    'pclass', 'sex_encoded', 'age_scaled', 'fare_log_scaled',
    'sibsp_scaled', 'parch_scaled', 'family_size', 'has_cabin_record',
    'embarked_Q', 'embarked_S', 'who_man', 'who_woman', 'is_child'
]

X = df[feature_cols]
y = df['survived']

print("\nFeature matrix shape:", X.shape)
print("\nFeature list:", feature_cols)
print("\nAny missing values in X?", X.isnull().sum().sum())
print("\nNew feature is_child distribution:\n", df['is_child'].value_counts())
print("Survival rate for is_child=1:", df[df['is_child']==1]['survived'].mean().round(3))
print("Survival rate for is_child=0:", df[df['is_child']==0]['survived'].mean().round(3))

X.to_csv('data/X_features.csv', index=False)
y.to_csv('data/y_target.csv', index=False)
