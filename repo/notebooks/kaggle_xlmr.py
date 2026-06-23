"""
=============================================================
XLM-RoBERTa Fine-Tuning: Crisis & Defense Misinformation Detection
=============================================================

HOW TO RUN ON KAGGLE:
1. Create new Kaggle notebook (GPU T4 x2, or P100)
2. Upload crisis_defense_benchmark_v2.csv as a Kaggle Dataset
3. Paste this entire script into a code cell
4. Update DATA_PATH below to match your dataset path
5. Run All -- takes ~25-35 minutes on free T4
6. Copy the printed results table and paste it back to Claude

WHAT THIS PRODUCES:
- Zero-shot XLM-R result  (no fine-tuning, baseline)
- Fine-tuned XLM-R result (3 epochs on crisis/defense data)
- Per-source breakdown (CoAID, LIAR, FakeNewsNet-PolitiFact)
- Per crisis-type breakdown (health_crisis, political_defense)
- Saves results to /kaggle/working/xlmr_results.json
=============================================================
"""

# ── Install / imports ─────────────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "transformers", "datasets", "accelerate", "-q"])

import os, json, warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
warnings.filterwarnings('ignore')

# ── CONFIG — update DATA_PATH to your file ────────────────────────────
DATA_PATH  = "/kaggle/input/YOUR-DATASET-NAME/crisis_defense_benchmark_v2.csv"
MODEL_NAME = "xlm-roberta-base"   # 278M params, fits on free T4
MAX_LEN    = 128                  # shorter = faster; 128 is fine for short news text
BATCH_SIZE = 32
EPOCHS     = 3
LR         = 2e-5
SEED       = 42
OUT_DIR    = "/kaggle/working"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Dataset class ─────────────────────────────────────────────────────
class MisinfoDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Load data ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} samples")
print(df.groupby(['source', 'label_name']).size())

X = df['text'].values
y = df['label'].values

X_train, X_test, y_train, y_test, idx_tr, idx_te = train_test_split(
    X, y, df.index, test_size=0.2, stratify=y, random_state=SEED)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED)

df_test = df.loc[idx_te].copy()
print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── Tokenizer ─────────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = MisinfoDataset(X_train, y_train, tokenizer, MAX_LEN)
val_ds   = MisinfoDataset(X_val,   y_val,   tokenizer, MAX_LEN)
test_ds  = MisinfoDataset(X_test,  y_test,  tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ── Evaluate helper ───────────────────────────────────────────────────
def evaluate(model, loader, return_preds=False):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labs = batch['label'].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
    p,r,f,_ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = None
    if return_preds:
        return p, r, f, acc, auc, np.array(all_preds)
    return p, r, f, acc, auc

# ── ZERO-SHOT evaluation (no fine-tuning) ─────────────────────────────
print("\n" + "="*60)
print("ZERO-SHOT XLM-R (no fine-tuning)")
print("="*60)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2).to(device)

p0, r0, f0, acc0, auc0 = evaluate(model, test_loader)
print(f"Accuracy:  {acc0*100:.2f}%")
print(f"Precision: {p0*100:.2f}%")
print(f"Recall:    {r0*100:.2f}%")
print(f"Macro-F1:  {f0*100:.2f}%")
print(f"ROC-AUC:   {auc0*100:.2f}%" if auc0 else "ROC-AUC:   N/A")

zero_shot = dict(
    accuracy=round(acc0*100,2), macro_precision=round(p0*100,2),
    macro_recall=round(r0*100,2), macro_f1=round(f0*100,2),
    roc_auc=round(auc0*100,2) if auc0 else 'N/A'
)

# ── FINE-TUNING ───────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"FINE-TUNING XLM-R — {EPOCHS} epochs")
print("="*60)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
loss_fn = torch.nn.CrossEntropyLoss()

best_val_f1, best_state = 0, None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labs = batch['label'].to(device)

        optimizer.zero_grad()
        out  = model(input_ids=ids, attention_mask=mask)
        loss = loss_fn(out.logits, labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % 20 == 0:
            print(f"  Epoch {epoch} Step {step+1}/{len(train_loader)} "
                  f"Loss: {total_loss/(step+1):.4f}")

    # Validation
    vp, vr, vf, vacc, vauc = evaluate(model, val_loader)
    print(f"\nEpoch {epoch} Val: Acc={vacc*100:.1f}%  F1={vf*100:.2f}%  "
          f"AUC={vauc*100:.2f}%" if vauc else
          f"\nEpoch {epoch} Val: Acc={vacc*100:.1f}%  F1={vf*100:.2f}%")

    if vf > best_val_f1:
        best_val_f1 = vf
        best_state  = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  ** New best val F1: {best_val_f1*100:.2f}%")

# ── FINAL TEST EVALUATION ─────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL TEST RESULTS — Fine-Tuned XLM-R")
print("="*60)

if best_state is not None:
    model.load_state_dict(best_state)

p1, r1, f1, acc1, auc1, test_preds = evaluate(model, test_loader, return_preds=True)
df_test = df_test.copy()
df_test['pred'] = test_preds

print(f"Accuracy:  {acc1*100:.2f}%")
print(f"Precision: {p1*100:.2f}%")
print(f"Recall:    {r1*100:.2f}%")
print(f"Macro-F1:  {f1*100:.2f}%")
print(f"ROC-AUC:   {auc1*100:.2f}%" if auc1 else "ROC-AUC:   N/A")

fine_tuned = dict(
    accuracy=round(acc1*100,2), macro_precision=round(p1*100,2),
    macro_recall=round(r1*100,2), macro_f1=round(f1*100,2),
    roc_auc=round(auc1*100,2) if auc1 else 'N/A'
)

# Per-source breakdown
print("\n--- Per-source breakdown ---")
source_breakdown = {}
for src in df_test['source'].unique():
    mask = df_test['source'] == src
    ys   = df_test.loc[mask, 'label'].values
    ps   = df_test.loc[mask, 'pred'].values
    if len(ys) < 5:
        continue
    sp,sr,sf,_ = precision_recall_fscore_support(ys, ps, average='macro')
    source_breakdown[src] = dict(n=int(mask.sum()),
        precision=round(sp*100,2), recall=round(sr*100,2), f1=round(sf*100,2))
    print(f"  {src:38}: P={sp*100:.1f}% R={sr*100:.1f}% F1={sf*100:.1f}%  (n={mask.sum()})")

# Per crisis-type breakdown
print("\n--- Per crisis-type breakdown ---")
crisis_breakdown = {}
for ct in df_test['crisis_type'].unique():
    mask = df_test['crisis_type'] == ct
    yc   = df_test.loc[mask, 'label'].values
    pc   = df_test.loc[mask, 'pred'].values
    if len(yc) < 5:
        continue
    cp,cr,cf,_ = precision_recall_fscore_support(yc, pc, average='macro')
    crisis_breakdown[ct] = dict(n=int(mask.sum()),
        precision=round(cp*100,2), recall=round(cr*100,2), f1=round(cf*100,2))
    print(f"  {ct:28}: P={cp*100:.1f}% R={cr*100:.1f}% F1={cf*100:.1f}%  (n={mask.sum()})")

# ── SUMMARY TABLE (copy this to Claude) ──────────────────────────────
print("\n" + "="*60)
print("COPY THIS OUTPUT TO CLAUDE:")
print("="*60)

summary = {
    "XLM-R Zero-Shot":  zero_shot,
    "XLM-R Fine-Tuned": fine_tuned,
    "per_source_finetuned":      source_breakdown,
    "per_crisis_type_finetuned": crisis_breakdown,
}
print(json.dumps(summary, indent=2))

# Save
torch.save(best_state, f"{OUT_DIR}/xlmr_best_model.pt")
with open(f"{OUT_DIR}/xlmr_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nModel saved to {OUT_DIR}/xlmr_best_model.pt")
print(f"Results saved to {OUT_DIR}/xlmr_results.json")
print("\nDone.")
