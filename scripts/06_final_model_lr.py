import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

plt.rcParams.update({'figure.dpi':150, 'font.size':10})

X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv'); y_test = pd.read_csv('data/y_test.csv').squeeze()
lr = joblib.load('data/best_lr_model.joblib')

y_pred = lr.predict(X_test)

# Confusion matrix for the FINAL selected model (Logistic Regression)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5.5,4.8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
            xticklabels=['Did not survive','Survived'], yticklabels=['Did not survive','Survived'],
            cbar_kws={'label':'Count'})
ax.set_title('Confusion Matrix — Final Model: Logistic Regression (Test Set)', fontsize=11.5, fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('figures/05_confusion_matrix_lr_final.png', bbox_inches='tight')
plt.close()
print(classification_report(y_test, y_pred, target_names=['Did not survive','Survived']))

# Logistic regression coefficients (interpretability — odds ratios)
coefs = pd.Series(lr.coef_[0], index=X_train.columns).sort_values()
fig, ax = plt.subplots(figsize=(7.5,5.5))
colors = ['#C44E52' if c < 0 else '#55A868' for c in coefs.values]
bars = ax.barh(coefs.index, coefs.values, color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Logistic Regression Coefficients\n(positive = increases survival odds, negative = decreases)', fontsize=12, fontweight='bold')
ax.set_xlabel('Coefficient (log-odds scale)')
for b, v in zip(bars, coefs.values):
    ax.text(v + (0.03 if v>=0 else -0.03), b.get_y()+b.get_height()/2, f"{v:.2f}",
            va='center', ha='left' if v>=0 else 'right', fontsize=8)
plt.tight_layout()
plt.savefig('figures/06_lr_coefficients.png', bbox_inches='tight')
plt.close()
print("\nCoefficients:\n", coefs.round(3))
