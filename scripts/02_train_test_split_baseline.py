import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

X = pd.read_csv('data/X_features.csv')
y = pd.read_csv('data/y_target.csv').squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train survival rate: {y_train.mean():.3f}, Test survival rate: {y_test.mean():.3f}")
# stratify=y keeps the 38.4% survival rate consistent between train and test splits

# Trivial baseline: always predict the majority class
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
print(f"\nDummy baseline (always predict 'did not survive') test accuracy: {dummy.score(X_test, y_test):.3f}")
print("This is the number any real model needs to beat convincingly.")

X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
