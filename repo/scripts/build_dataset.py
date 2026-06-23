"""
Build the combined three-dataset crisis & defense benchmark.
Sources:
  1. CoAID       — COVID-19 health crisis        (crisis_type: health_crisis)
  2. LIAR        — Political / gov disinformation (crisis_type: political_defense)
  3. FakeNewsNet — PolitiFact fake/real news      (crisis_type: political_defense)
"""
import pandas as pd
import numpy as np
import glob, os, warnings
warnings.filterwarnings('ignore')

OUT = "/home/claude/results"
os.makedirs(OUT, exist_ok=True)

all_dfs = []

# ── 1. CoAID ──────────────────────────────────────────────────────────
print("Loading CoAID...")
for fpath in sorted(glob.glob("/home/claude/data/coaid/*.csv")):
    fname = os.path.basename(fpath)
    label = 0 if "Fake" in fname else 1
    try:
        df = pd.read_csv(fpath, encoding='utf-8-sig', on_bad_lines='skip')
    except:
        continue
    text_col = next((c for c in ['content','abstract','title','newstitle']
                     if c in df.columns), None)
    if text_col is None:
        continue
    df['text']        = df[text_col].fillna('').astype(str)
    df['label']       = label
    df['label_name']  = 'fake' if label == 0 else 'real'
    df['crisis_type'] = 'health_crisis'
    df['source']      = 'CoAID'
    all_dfs.append(df[['text','label','label_name','crisis_type','source']])

coaid_raw = pd.concat(all_dfs, ignore_index=True)
coaid_raw = coaid_raw[coaid_raw['text'].str.len() > 30]
print(f"  CoAID raw: {len(coaid_raw)} | fake={sum(coaid_raw.label==0)} real={sum(coaid_raw.label==1)}")

# ── 2. LIAR ───────────────────────────────────────────────────────────
print("Loading LIAR...")
liar_dfs = []
# Columns: id, label, statement, subject, speaker, job, state, party,
#          barely_true_counts, false_counts, half_true_counts,
#          mostly_true_counts, pants_fire_counts, context
LIAR_COLS = ['id','veracity','statement','subject','speaker',
             'job','state','party','barely_true','false_count',
             'half_true','mostly_true','pants_fire','context']

# Map 6-class labels to binary: fake = false|pants-fire|barely-true; real = true|mostly-true|half-true
FAKE_LABELS  = {'false', 'pants-fire', 'barely-true'}
REAL_LABELS  = {'true', 'mostly-true', 'half-true'}

for split in ['train','test','valid']:
    fpath = f"/home/claude/data/liar/{split}.tsv"
    if not os.path.exists(fpath):
        continue
    df = pd.read_csv(fpath, sep='\t', header=None,
                     names=LIAR_COLS[:len(open(fpath).readline().split('\t'))],
                     on_bad_lines='skip')
    if 'veracity' not in df.columns or 'statement' not in df.columns:
        # fallback: first two columns
        df = pd.read_csv(fpath, sep='\t', header=None, on_bad_lines='skip')
        df.columns = ['id','veracity'] + [f'c{i}' for i in range(len(df.columns)-2)]
        df['statement'] = df.apply(lambda r: ' '.join(str(v) for v in r.iloc[2:5]
                                   if pd.notna(v)), axis=1)

    df['veracity'] = df['veracity'].str.strip().str.lower()
    df_fake = df[df['veracity'].isin(FAKE_LABELS)].copy()
    df_real = df[df['veracity'].isin(REAL_LABELS)].copy()
    for sub, lbl in [(df_fake, 0), (df_real, 1)]:
        sub = sub.copy()
        sub['text']        = sub['statement'].fillna('').astype(str)
        sub['label']       = lbl
        sub['label_name']  = 'fake' if lbl == 0 else 'real'
        sub['crisis_type'] = 'political_defense'
        sub['source']      = 'LIAR'
        liar_dfs.append(sub[['text','label','label_name','crisis_type','source']])

liar_raw = pd.concat(liar_dfs, ignore_index=True) if liar_dfs else pd.DataFrame()
liar_raw = liar_raw[liar_raw['text'].str.len() > 15]
print(f"  LIAR raw: {len(liar_raw)} | fake={sum(liar_raw.label==0)} real={sum(liar_raw.label==1)}")

# ── 3. FakeNewsNet (PolitiFact only — political/defense relevant) ─────
print("Loading FakeNewsNet (PolitiFact)...")
fnn_dfs = []
for fname, lbl in [('politifact_fake', 0), ('politifact_real', 1)]:
    fpath = f"/home/claude/data/fakenewsnet/{fname}.csv"
    if not os.path.exists(fpath):
        continue
    df = pd.read_csv(fpath, on_bad_lines='skip')
    # PolitiFact has: id, news_url, title, tweet_ids
    if 'title' in df.columns:
        df['text'] = df['title'].fillna('').astype(str)
    else:
        continue
    df['label']       = lbl
    df['label_name']  = 'fake' if lbl == 0 else 'real'
    df['crisis_type'] = 'political_defense'
    df['source']      = 'FakeNewsNet-PolitiFact'
    fnn_dfs.append(df[['text','label','label_name','crisis_type','source']])

fnn_raw = pd.concat(fnn_dfs, ignore_index=True) if fnn_dfs else pd.DataFrame()
fnn_raw = fnn_raw[fnn_raw['text'].str.len() > 10]
print(f"  FakeNewsNet-PolitiFact raw: {len(fnn_raw)} | fake={sum(fnn_raw.label==0)} real={sum(fnn_raw.label==1)}")

# ── 4. Combine, clean, balance per source ────────────────────────────
print("\nBuilding combined benchmark...")

def balance_source(df, max_per_class=600):
    fake = df[df['label']==0]
    real = df[df['label']==1]
    n = min(len(fake), len(real), max_per_class)
    return pd.concat([
        fake.sample(n, random_state=42),
        real.sample(n, random_state=42)
    ])

coaid_b = balance_source(coaid_raw, 400)
liar_b  = balance_source(liar_raw,  600)
fnn_b   = balance_source(fnn_raw,   400)

combined = pd.concat([coaid_b, liar_b, fnn_b], ignore_index=True)
combined = combined[combined['text'].str.len() > 15]
combined = combined.drop_duplicates(subset='text').reset_index(drop=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal combined benchmark: {len(combined)} samples")
print(combined.groupby(['source','label_name']).size().to_string())
print(f"\nOverall: fake={sum(combined.label==0)} real={sum(combined.label==1)}")
print(f"Crisis types: {combined['crisis_type'].value_counts().to_dict()}")

combined.to_csv(f"{OUT}/crisis_defense_benchmark_v2.csv", index=False)
print(f"\nSaved: crisis_defense_benchmark_v2.csv")
