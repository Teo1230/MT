# ================================
# 0) Importuri
# ================================
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd

# ================================
# 1) Ia un subset din Text2Emoji (engleză → emoji)
# ================================

MAX_SAMPLES = 20000
SEED = 42

ds = load_dataset("KomeijiForce/Text2Emoji")["train"]

# Shuffle + subset
ds = ds.shuffle(seed=SEED).select(range(MAX_SAMPLES))

# Filtrări obligatorii
ds = ds.filter(lambda ex: ex["emoji"] is not None and ex["emoji"].strip() != "")
ds = ds.filter(lambda ex: ex["text"] is not None and ex["text"].strip() != "")

print("După filtrare:", ds)
print(ds[0])

# ================================
# 2) Traducerea ENG → RO cu Helsinki-NLP (Safe)
# ================================

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ro"
MAX_LENGTH = 128

tkn_mt = AutoTokenizer.from_pretrained(MODEL_NAME)
mt = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

translator = pipeline(
    "translation",
    model=mt,
    tokenizer=tkn_mt,
    truncation=True,
    max_length=MAX_LENGTH,
)

# Funcție de traducere sigură (fără None / blank)
def en2ro_batch(batch):
    texts = []
    for t in batch["text"]:
        if t is None or t.strip() == "":
            texts.append("PLACEHOLDER")  
        else:
            texts.append(t.strip())

    ro = translator(texts)
    batch["ro"] = [x["translation_text"] for x in ro]
    return batch

ds_ro = ds.map(en2ro_batch, batched=True, batch_size=32)

# Păstrăm doar ro + emoji
ds_ro = ds_ro.remove_columns(
    [c for c in ds_ro.column_names if c not in ["ro", "emoji"]]
)

print(ds_ro[0])

# ================================
# 3) Curățare minimă + eliminare duplicate + split
# ================================

def clean_batch(batch):
    ro_clean = []
    emoji_clean = []

    for txt, emo in zip(batch["ro"], batch["emoji"]):
        if txt is None or emo is None:
            continue

        txt = " ".join(txt.strip().split())
        emo = emo.strip()

        if txt != "" and emo != "":
            ro_clean.append(txt)
            emoji_clean.append(emo)

    return {"ro": ro_clean, "emoji": emoji_clean}

ds_ro_clean = ds_ro.map(clean_batch, batched=True, remove_columns=["ro", "emoji"])

# Eliminare duplicate (folosind pandas → Dataset)
df = ds_ro_clean.to_pandas().drop_duplicates(subset=["ro", "emoji"])
ds_ro_clean = Dataset.from_pandas(df, preserve_index=False)

print("După eliminarea duplicatelor:", ds_ro_clean)

# Split (80/10/10)
splits = ds_ro_clean.train_test_split(test_size=0.2, seed=SEED)
test_valid = splits["test"].train_test_split(test_size=0.5, seed=SEED)

dataset = DatasetDict(
    {
        "train": splits["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"],
    }
)

print(dataset)

# ================================
# 4) Salvarea datelor
# ================================

SAVE_DIR = "text2emoji_ro_dataset"
dataset.save_to_disk(SAVE_DIR)
print(f"Am salvat datasetul HuggingFace în: {SAVE_DIR}/")

dataset["train"].to_pandas().to_csv("text2emoji_ro_train.csv", index=False)
dataset["validation"].to_pandas().to_csv("text2emoji_ro_valid.csv", index=False)
dataset["test"].to_pandas().to_csv("text2emoji_ro_test.csv", index=False)

print("CSV-urile au fost salvate: text2emoji_ro_[train|valid|test].csv")
