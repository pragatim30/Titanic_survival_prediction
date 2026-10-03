import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Tune Random Forest ---
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, None],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid,
                        cv=cv, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train, y_train)
print("Random Forest best params:", rf_grid.best_params_)
print("Random Forest best CV F1:", round(rf_grid.best_score_, 3))

# --- Tune Logistic Regression ---
lr_param_grid = {
    'C': [0.01, 0.1, 0.5, 1, 5, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
}
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), lr_param_grid,
                        cv=cv, scoring='f1', n_jobs=-1)
lr_grid.fit(X_train, y_train)
print("\nLogistic Regression best params:", lr_grid.best_params_)
print("Logistic Regression best CV F1:", round(lr_grid.best_score_, 3))

joblib.dump(rf_grid.best_estimator_, 'data/best_rf_model.joblib')
joblib.dump(lr_grid.best_estimator_, 'data/best_lr_model.joblib')
print("\nSaved both tuned models.")
