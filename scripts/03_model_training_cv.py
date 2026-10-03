import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM (RBF kernel)': SVC(probability=True, random_state=42),
}

results = {}
for name, model in models.items():
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    results[name] = {m: cv_results[f'test_{m}'].mean() for m in scoring}
    results[name]['std_accuracy'] = cv_results['test_accuracy'].std()
    print(f"\n{name}:")
    for m in scoring:
        print(f"  {m}: {cv_results[f'test_{m}'].mean():.3f} (+/- {cv_results[f'test_{m}'].std():.3f})")

results_df = pd.DataFrame(results).T.round(3)
results_df.to_csv('data/cv_model_comparison.csv')
print("\n\nSummary table:\n", results_df)
