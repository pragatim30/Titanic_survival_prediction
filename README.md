# Titanic Survival Prediction — Supervised Learning (Week 4)

Binary classification task predicting Titanic passenger survival, built on top
of the Week 1 (cleaning) and Week 2 (EDA/feature engineering) work in this
same project series.

## Structure

```
data/
  titanic_eda_features.csv   # input (from Week 2)
  cv_model_comparison.csv    # 5-fold CV results for all 3 candidate models
  test_predictions.csv       # final model predictions on the held-out test set
scripts/
  01_problem_setup_and_features.py   # feature set assembly + new is_child feature
  02_train_test_split_baseline.py    # stratified split + dummy baseline
  03_model_training_cv.py            # 5-fold CV: Logistic Regression, Random Forest, SVM
  04_hyperparameter_tuning.py        # GridSearchCV for RF and Logistic Regression
  05_final_evaluation.py             # test-set evaluation, ROC/PR curves, RF feature importance
  06_final_model_lr.py               # final model (Logistic Regression) confusion matrix + coefficients
  07_cv_comparison_chart.py          # CV comparison bar chart
figures/     # all 7 charts used in the report
models/      # saved, tuned model files (.joblib)
report/
  Supervised_Learning_Report.docx
```

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python scripts/01_problem_setup_and_features.py
python scripts/02_train_test_split_baseline.py
python scripts/03_model_training_cv.py
python scripts/04_hyperparameter_tuning.py
python scripts/05_final_evaluation.py
python scripts/06_final_model_lr.py
python scripts/07_cv_comparison_chart.py
```

## Key result worth knowing before reading the report

Random Forest won on 5-fold cross-validation (F1 = 0.765 vs Logistic
Regression's 0.757), but on the held-out test set Logistic Regression
outperformed it on every metric (82.7% vs 82.1% accuracy, 0.853 vs 0.847
ROC-AUC). Logistic Regression was selected as the final model based on the
test-set result, not the CV result — the report discusses this reversal
directly rather than only reporting whichever model looked best.
