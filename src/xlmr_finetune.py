"""
xlmr_finetune.py
────────────────
XLM-RoBERTa fine-tuning for the crisis benchmark.
Designed to run on Kaggle T4 GPU (free tier).

Hardware:   Kaggle T4 GPU (~16 GB VRAM)
Runtime:    ~18 min/epoch, ~55 min total (3 epochs)
Best result: 81.70% Macro-F1, 92.04% ROC-AUC (epoch 3, benchmark v4)

Usage:
    # On Kaggle: upload crisis_defense_benchmark_v4.csv as a dataset,
    # then run this notebook cell:
    python xlmr_finetune.py --data /kaggle/input/crisis-bench/crisis_defense_benchmark_v4.csv \
                            --output /kaggle/working/xlmr_results.json \
                            --epochs 3

Requirements:
    transformers>=4.30, torch>=2.0, scikit-learn, pandas, numpy
"""

import json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


MODEL_NAME = "xlm-roberta-base"
MAX_LEN    = 128
BATCH_SIZE = 32
LR         = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
SEED       = 42


class CrisisDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["label"].to(device)
            out  = model(ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            preds = torch.argmax(out.logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labs.cpu().numpy())

    return {
        "accuracy":  round(accuracy_score(all_labels, all_preds), 4),
        "macro_f1":  round(f1_score(all_labels, all_preds, average="macro"), 4),
        "roc_auc":   round(roc_auc_score(all_labels, all_probs), 4),
    }


def run_finetuning(
    data_path: str,
    output_path: str = "results/xlmr_results.json",
    epochs: int = 3,
    seed: int = SEED,
) -> dict:

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(data_path).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.2, stratify=df["label"], random_state=seed,
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    model     = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    train_ds = CrisisDataset(X_train, y_train, tokenizer)
    val_ds   = CrisisDataset(X_val,   y_val,   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dl) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Zero-shot baseline
    print("\nEpoch 0 (zero-shot):")
    zero_shot = evaluate_model(model, val_dl, device)
    print(f"  {zero_shot}")

    history = [{"epoch": 0, **zero_shot}]

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_dl:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["label"].to(device)
            optimizer.zero_grad()
            out  = model(ids, attention_mask=mask, labels=labs)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()

        avg_loss = total_loss / len(train_dl)
        metrics  = evaluate_model(model, val_dl, device)
        metrics["avg_loss"] = round(avg_loss, 4)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}  {metrics}")
        history.append({"epoch": epoch, **metrics})

    # Save model checkpoint
    model.save_pretrained("results/xlmr_checkpoint")
    tokenizer.save_pretrained("results/xlmr_checkpoint")

    results = {
        "model": MODEL_NAME,
        "config": {"epochs": epochs, "lr": LR, "batch_size": BATCH_SIZE,
                   "max_len": MAX_LEN, "weight_decay": WEIGHT_DECAY},
        "history": history,
        "best_epoch": max(history[1:], key=lambda x: x["macro_f1"]),
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/xlmr_results.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_finetuning(args.data, args.output, args.epochs, args.seed)
