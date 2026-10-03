import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                              precision_recall_curve, accuracy_score, f1_score, roc_auc_score)

plt.rcParams.update({'figure.dpi':150, 'font.size':10})

X_train = pd.read_csv('data/X_train.csv'); y_train = pd.read_csv('data/y_train.csv').squeeze()
X_test = pd.read_csv('data/X_test.csv'); y_test = pd.read_csv('data/y_test.csv').squeeze()

rf = joblib.load('data/best_rf_model.joblib')
lr = joblib.load('data/best_lr_model.joblib')

# Final model: Random Forest (slightly higher CV F1 + gives feature importance)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
print(f"FINAL MODEL (Random Forest, tuned) — Held-out Test Set:")
print(f"Accuracy: {acc:.3f}")
print(f"F1: {f1:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")
print("\nFull classification report:\n", classification_report(y_test, y_pred, target_names=['Did not survive','Survived']))

# Also test Logistic Regression on same test set for final comparison
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:,1]
print(f"\nComparison — Logistic Regression (tuned) — Test accuracy: {accuracy_score(y_test, y_pred_lr):.3f}, "
      f"F1: {f1_score(y_test, y_pred_lr):.3f}, ROC-AUC: {roc_auc_score(y_test, y_proba_lr):.3f}")

# ---- Confusion matrix ----
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5.5,4.8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Did not survive','Survived'], yticklabels=['Did not survive','Survived'],
            cbar_kws={'label':'Count'})
ax.set_title('Confusion Matrix — Random Forest (Test Set)', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('figures/01_confusion_matrix.png', bbox_inches='tight')
plt.close()

# ---- ROC curves for both models ----
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fig, ax = plt.subplots(figsize=(6.5,5.5))
ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc(fpr_rf,tpr_rf):.3f})', color='#4C72B0', linewidth=2)
ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc(fpr_lr,tpr_lr):.3f})', color='#DD8452', linewidth=2)
ax.plot([0,1],[0,1], linestyle='--', color='gray', label='Random guess (AUC=0.5)')
ax.set_title('ROC Curve — Model Comparison (Test Set)', fontsize=13, fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('figures/02_roc_curve.png', bbox_inches='tight')
plt.close()

# ---- Precision-Recall curve ----
prec, rec, _ = precision_recall_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6.5,5))
ax.plot(rec, prec, color='#55A868', linewidth=2)
ax.axhline(y_test.mean(), color='gray', linestyle='--', label=f'Baseline (prevalence={y_test.mean():.2f})')
ax.set_title('Precision-Recall Curve — Random Forest (Test Set)', fontsize=13, fontweight='bold')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend()
plt.tight_layout()
plt.savefig('figures/03_precision_recall_curve.png', bbox_inches='tight')
plt.close()

# ---- Feature importance ----
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7.5,5.5))
bars = ax.barh(importances.index, importances.values, color='#4C72B0')
ax.set_title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance (Gini-based)')
for b, v in zip(bars, importances.values):
    ax.text(v+0.002, b.get_y()+b.get_height()/2, f"{v:.3f}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig('figures/04_feature_importance.png', bbox_inches='tight')
plt.close()

pd.DataFrame({'y_test':y_test, 'y_pred':y_pred, 'y_proba':y_proba}).to_csv('data/test_predictions.csv', index=False)
print("\nSaved all evaluation figures.")
