import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'figure.dpi':150, 'font.size':10})
df = pd.read_csv('data/cv_model_comparison.csv', index_col=0)
print(df)

metrics = ['accuracy','precision','recall','f1','roc_auc']
fig, ax = plt.subplots(figsize=(10,5.5))
x = np.arange(len(metrics))
width = 0.25
colors = ['#4C72B0','#DD8452','#55A868']
for i, model in enumerate(df.index):
    vals = df.loc[model, metrics].values
    bars = ax.bar(x + i*width, vals, width, label=model, color=colors[i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.3f}", ha='center', fontsize=7.5, rotation=90)
ax.set_xticks(x + width)
ax.set_xticklabels(['Accuracy','Precision','Recall','F1','ROC-AUC'])
ax.set_ylim(0, 1.05)
ax.set_title('5-Fold Cross-Validation: Model Comparison (Training Set)', fontsize=13, fontweight='bold')
ax.set_ylabel('Score')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('figures/07_cv_model_comparison.png', bbox_inches='tight')
plt.close()
print("saved")
