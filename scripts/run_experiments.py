"""
Full experiment pipeline on the 3-dataset combined benchmark.
Produces all tables and figures for the paper.
"""
import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                              confusion_matrix, classification_report)
import shap

OUT = "/home/claude/results"
os.makedirs(OUT, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────
df = pd.read_csv(f"{OUT}/crisis_defense_benchmark_v2.csv")
print(f"Loaded {len(df)} samples | sources: {df['source'].unique()}")

X = df['text'].values
y = df['label'].values

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, stratify=y, random_state=42)

df_test = df.loc[idx_test].copy()

# ── Models ────────────────────────────────────────────────────────────
models = {
    "TF-IDF + Logistic Regression": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=25000, ngram_range=(1,2), sublinear_tf=True)),
        ('clf',   LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ]),
    "TF-IDF + Linear SVM": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=25000, ngram_range=(1,2), sublinear_tf=True)),
        ('clf',   LinearSVC(max_iter=3000, C=1.0, random_state=42))
    ]),
    "TF-IDF + Random Forest": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1,1), sublinear_tf=True)),
        ('clf',   RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42))
    ]),
    "TF-IDF + Gradient Boosting": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1,1), sublinear_tf=True)),
        ('clf',   GradientBoostingClassifier(n_estimators=150, random_state=42))
    ]),
}

short = {
    "TF-IDF + Logistic Regression": "LR",
    "TF-IDF + Linear SVM":          "SVM",
    "TF-IDF + Random Forest":       "RF",
    "TF-IDF + Gradient Boosting":   "GBM",
}

results = {}
print("\n" + "="*65)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    p,r,f,_ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    p0,r0,f0,_ = precision_recall_fscore_support(y_test, y_pred, labels=[0], average='macro')
    p1,r1,f1,_ = precision_recall_fscore_support(y_test, y_pred, labels=[1], average='macro')
    acc = (y_pred == y_test).mean()
    try:
        if hasattr(model, 'predict_proba'):
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        else:
            auc = None
    except:
        auc = None
    results[name] = dict(
        accuracy=round(acc*100,2), macro_precision=round(p*100,2),
        macro_recall=round(r*100,2), macro_f1=round(f*100,2),
        fake_f1=round(f0*100,2), real_f1=round(f1*100,2),
        roc_auc=round(auc*100,2) if auc else 'N/A',
        model_obj=model, y_pred=y_pred
    )
    print(f"{short[name]:5} | Acc {acc*100:.1f}% | P {p*100:.1f}% | R {r*100:.1f}% | F1 {f*100:.1f}%  ROC:{auc*100:.1f}%" if auc else f"{short[name]:5} | Acc {acc*100:.1f}% | P {p*100:.1f}% | R {r*100:.1f}% | F1 {f*100:.1f}%")

# ── Per-source breakdown ──────────────────────────────────────────────
print("\n--- Per-source breakdown (best model: SVM) ---")
best_model = models["TF-IDF + Linear SVM"]
best_pred  = results["TF-IDF + Linear SVM"]["y_pred"]

source_results = {}
for src in df_test['source'].unique():
    mask = df_test['source'] == src
    y_s = y_test[mask]
    p_s = best_pred[mask]
    if len(y_s) < 5:
        continue
    p_,r_,f_,_ = precision_recall_fscore_support(y_s, p_s, average='macro')
    source_results[src] = dict(n=len(y_s),
        precision=round(p_*100,2), recall=round(r_*100,2), f1=round(f_*100,2))
    print(f"  {src:35}: P={p_*100:.1f}% R={r_*100:.1f}% F1={f_*100:.1f}%  (n={len(y_s)})")

# Per crisis-type breakdown
print("\n--- Per crisis-type breakdown (best model: SVM) ---")
crisis_results = {}
for ct in df_test['crisis_type'].unique():
    mask = df_test['crisis_type'] == ct
    y_c = y_test[mask]
    p_c = best_pred[mask]
    if len(y_c) < 5:
        continue
    p_,r_,f_,_ = precision_recall_fscore_support(y_c, p_c, average='macro')
    crisis_results[ct] = dict(n=len(y_c),
        precision=round(p_*100,2), recall=round(r_*100,2), f1=round(f_*100,2))
    print(f"  {ct:25}: P={p_*100:.1f}% R={r_*100:.1f}% F1={f_*100:.1f}%  (n={len(y_c)})")

# Save JSON
clean = {k: {kk:vv for kk,vv in v.items() if kk not in ('model_obj','y_pred')}
         for k,v in results.items()}
clean['per_source']       = source_results
clean['per_crisis_type']  = crisis_results
with open(f"{OUT}/results_combined.json","w") as f:
    json.dump(clean, f, indent=2)

# ── TABLE 2: Main results ─────────────────────────────────────────────
rows = []
for mn, r in results.items():
    rows.append([short[mn],
                 f"{r['accuracy']:.2f}%",
                 f"{r['macro_precision']:.2f}%",
                 f"{r['macro_recall']:.2f}%",
                 f"{r['macro_f1']:.2f}%",
                 f"{r['fake_f1']:.2f}%",
                 f"{r['real_f1']:.2f}%",
                 str(r['roc_auc'])+'%' if r['roc_auc']!='N/A' else 'N/A'])

col_labels = ['Model','Acc.','Precision','Recall','Macro-F1','Fake F1','Real F1','ROC-AUC']
fig, ax = plt.subplots(figsize=(16, 3.4))
ax.axis('off')
tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.9)
f1_vals = [results[m]['macro_f1'] for m in results]
best_row = f1_vals.index(max(f1_vals)) + 1
for j in range(len(col_labels)):
    tbl[(0,j)].set_facecolor('#1565C0')
    tbl[(0,j)].set_text_props(color='white', fontweight='bold')
    tbl[(best_row,j)].set_facecolor('#E8F5E9')
    tbl[(best_row,j)].set_text_props(fontweight='bold')
plt.title("Table 2: Baseline Results — Combined Crisis & Defense Benchmark\n"
          "(CoAID Health Crisis + LIAR Political + FakeNewsNet-PolitiFact)",
          fontsize=12, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(f"{OUT}/table2_combined_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved table2_combined_results.png")

# ── FIGURE 5: Per-source F1 heatmap ──────────────────────────────────
src_names = list(source_results.keys())
metrics_s = ['precision','recall','f1']
heat_data = np.array([[source_results[s][m] for m in metrics_s] for s in src_names])
short_src = [s.replace('FakeNewsNet-PolitiFact','FNN-PolitiFact') for s in src_names]

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(heat_data, cmap='YlGn', aspect='auto', vmin=55, vmax=100)
ax.set_xticks(range(3))
ax.set_xticklabels(['Precision','Recall','F1-Score'], fontsize=12)
ax.set_yticks(range(len(short_src)))
ax.set_yticklabels(short_src, fontsize=11)
for i in range(len(src_names)):
    for j in range(3):
        ax.text(j, i, f"{heat_data[i,j]:.1f}%", ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white' if heat_data[i,j] > 80 else 'black')
plt.colorbar(im, ax=ax, label='Score (%)')
ax.set_title("Figure 5: Per-Dataset Performance (TF-IDF + Linear SVM)\n"
             "Crisis & Defense Benchmark", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_per_source_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_per_source_heatmap.png")

# ── FIGURE 6: SHAP on combined benchmark ─────────────────────────────
print("\nRunning SHAP...")
lr = models["TF-IDF + Logistic Regression"]
vec = lr.named_steps['tfidf']
clf = lr.named_steps['clf']
X_test_tfidf = vec.transform(X_test)
explainer    = shap.LinearExplainer(clf, X_test_tfidf,
                                     feature_perturbation="interventional")
shap_vals    = explainer.shap_values(X_test_tfidf)
fnames       = vec.get_feature_names_out()
mean_signed  = shap_vals.mean(axis=0)

fake_idx = np.argsort(mean_signed)[:20]
real_idx = np.argsort(mean_signed)[::-1][:20]
fake_words = [(fnames[i], abs(mean_signed[i])) for i in fake_idx if len(fnames[i])>2][:15]
real_words = [(fnames[i], mean_signed[i])      for i in real_idx if len(fnames[i])>2][:15]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax2, words, color, title in [
    (axes[0], fake_words, '#EF5350',
     'Top Tokens: FAKE Misinformation\n(Crisis & Defense Benchmark)'),
    (axes[1], real_words, '#43A047',
     'Top Tokens: REAL Credible Content\n(Crisis & Defense Benchmark)')]:
    names_ = [w for w,v in words]
    vals_  = [v for w,v in words]
    ax2.barh(names_[::-1], vals_[::-1], color=color, alpha=0.85)
    ax2.set_xlabel('Mean |SHAP Value|', fontsize=11)
    ax2.set_title(title, fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

plt.suptitle("Figure 6: SHAP Explainability — Crisis & Defense Misinformation Detection\n"
             "Token-level attribution across health crisis + political defense domains",
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_shap_combined.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_shap_combined.png")

print("\nAll combined experiments complete.")
print(f"Best model: TF-IDF + Linear SVM | Macro-F1 = {results['TF-IDF + Linear SVM']['macro_f1']}%")
