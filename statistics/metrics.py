
import pandas as pd
import numpy as np
from Levenshtein import distance as lev_distance
import evaluate
from collections import Counter

# ===============================
# 1) LOAD CSV
# ===============================
CSV_PATH = "C:\\Users\Carla\Desktop\MT\\ex4.2\predictions\\test_predictions.csv"
path_output = '.\\evaluation_var2_4.2.csv'
df = pd.read_csv(CSV_PATH)

df["true"] = df["true"].astype(str).str.strip()
df["pred_default"] = df["pred_default"].astype(str).str.strip()
##############
gold = df["true"].tolist()
pred = df["pred_default"].tolist()

emoji_list_pred = [e for seq in df["pred_default"].astype(str) for e in seq]
unique_pred = set(emoji_list_pred)

print("Emoji unice în predicții:", len(unique_pred))
print(unique_pred)

# ===============================
# 2) METRIC FUNCTIONS
# ===============================

def exact_match(p, g):
    return int(p == g)

def emoji_f1(pred, gold):
    ps, gs = list(pred), list(gold)
    if not ps and not gs: return 1.0
    if not ps or not gs: return 0.0
    tp = sum(min(ps.count(c), gs.count(c)) for c in set(ps + gs))
    prec = tp / len(ps)
    rec = tp / len(gs)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

def jaccard(pred, gold):
    ps, gs = set(pred), set(gold)
    return len(ps & gs) / len(ps | gs) if ps | gs else 0.0

bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

# ===============================
# 3) COMPUTE PER-ROW METRICS
# ===============================

rows = []

for p, g in zip(pred, gold):
    rows.append({
        "true": g,
        "pred": p,
        "exact_match": exact_match(p, g),
        "emoji_f1": emoji_f1(p, g),
        "jaccard": jaccard(p, g),
        "levenshtein": lev_distance(p, g),
        "len_true": len(g),
        "len_pred": len(p),
        "len_diff": abs(len(g) - len(p))
    })

results_df = pd.DataFrame(rows)

# ===============================
# 4) GLOBAL METRICS
# ===============================

bleu_score = bleu.compute(predictions=pred, references=[[g] for g in gold])["score"]
chrf_score = chrf.compute(predictions=pred, references=[[g] for g in gold])["score"]

global_scores = {
    "Exact Match": results_df["exact_match"].mean(),
    "Emoji F1": results_df["emoji_f1"].mean(),
    "Jaccard": results_df["jaccard"].mean(),
    "Avg Levenshtein": results_df["levenshtein"].mean(),
    "Avg Length Diff": results_df["len_diff"].mean(),
    "BLEU": bleu_score,
    "ChrF": chrf_score
}

# ===============================
# 5) ADVANCED GLOBAL STATISTICS
# ===============================

# Cele mai frecvente 20 emoji în adevăr
true_flat = [c for seq in gold for c in seq]
pred_flat = [c for seq in pred for c in seq]

true_counts = Counter(true_flat).most_common(20)
pred_counts = Counter(pred_flat).most_common(20)

# Distribuția lungimilor
len_stats = {
    "Avg True Len": np.mean([len(t) for t in gold]),
    "Avg Pred Len": np.mean([len(p) for p in pred]),
    "Std True Len": np.std([len(t) for t in gold]),
    "Std Pred Len": np.std([len(p) for p in pred]),
}

# Rata de under- și over-generation
under = np.mean([1 if len(p) < len(g) else 0 for p, g in zip(pred, gold)])
over = np.mean([1 if len(p) > len(g) else 0 for p, g in zip(pred, gold)])
equal = 1 - under - over

gen_stats = {
    "Under-generate": under,
    "Over-generate": over,
    "Equal length": equal
}

# ===============================
# 6) PRINT RESULTS
# ===============================

print("\n===============================")
print("        GLOBAL METRICI")
print("===============================\n")

for k, v in global_scores.items():
    print(f"{k:20s}: {v:.4f}")

print("\n===============================")
print("        STATISTICI AVANSATE")
print("===============================\n")

print("Lungimi predicții:")
for k, v in len_stats.items():
    print(f"{k:20s}: {v:.4f}")

print("\nUnder/Over-generation:")
for k, v in gen_stats.items():
    print(f"{k:20s}: {v:.4f}")

print("\nTop 20 emoji în TRUE:")
print(true_counts)

print("\nTop 20 emoji în PRED:")
print(pred_counts)

# ===============================
# 7) SAVE RESULTS
# ===============================
results_df.to_csv(path_output, index=False)
