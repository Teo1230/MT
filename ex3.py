#!/usr/bin/env python3

import os
import json
from pathlib import Path

import pandas as pd
import sentencepiece as spm
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from transformers import (
    T5TokenizerFast,
    MT5Config,
    MT5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =====================================================================
# CONFIG
# =====================================================================
BASE_DIR = Path("mt5_emoji")
TOKENIZER_DIR = BASE_DIR / "tokenizer"
MODEL_DIR = BASE_DIR / "model"
REPORTS_DIR = BASE_DIR / "reports"
PLOTS_DIR = BASE_DIR / "plots"
SAMPLES_DIR = BASE_DIR / "samples"
LOGS_DIR = BASE_DIR / "logs"

for d in [BASE_DIR, TOKENIZER_DIR, MODEL_DIR, REPORTS_DIR, PLOTS_DIR, SAMPLES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = Path("text2emoji_ro_train.csv")
VALID_CSV = Path("text2emoji_ro_valid.csv")
TEST_CSV  = Path("text2emoji_ro_test.csv")

CORPUS_PATH = TOKENIZER_DIR / "train_corpus.txt"
SP_MODEL_PREFIX = TOKENIZER_DIR / "emoji_sp"
SP_MODEL_PATH = TOKENIZER_DIR / "emoji_sp.model"

BASE_MT5_NAME = "google/mt5-small"
VOCAB_SIZE = 16000
MAX_INPUT_LEN = 64
MAX_TARGET_LEN = 8


# =====================================================================
# CSV LOADING
# =====================================================================
def load_csvs():
    print("=== ÎNCĂRC CSV-URILE ===")

    train_df = pd.read_csv(TRAIN_CSV)
    valid_df = pd.read_csv(VALID_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    print(f"[INFO] Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    return train_df, valid_df, test_df


# =====================================================================
# CORPUS
# =====================================================================
def build_corpus(train_df, valid_df, test_df):
    print("=== CONSTRUIESC CORPUSUL ===")

    lines = []
    for df in [train_df, valid_df, test_df]:
        lines.extend(df["ro"].astype(str).tolist())
        lines.extend(df["emoji"].astype(str).tolist())

    with CORPUS_PATH.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.replace("\n", " ") + "\n")

    print(f"[OK] Corpus creat → {CORPUS_PATH} ({len(lines)} linii).")


# =====================================================================
# TOKENIZER SentencePiece
# =====================================================================
def build_tokenizer():
    print("=== CONSTRUIESC TOKENIZER ===")

    if SP_MODEL_PATH.exists():
        print(f"[INFO] Tokenizer SP existent, sar peste: {SP_MODEL_PATH}")
        return

    spm.SentencePieceTrainer.Train(
        input=str(CORPUS_PATH),
        model_prefix=str(SP_MODEL_PREFIX),
        vocab_size=VOCAB_SIZE,
        character_coverage=1.0,
        model_type="unigram",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    print("[OK] Tokenizer SP creat.")


# =====================================================================
# TOKENIZER HF + MT5 MODEL
# =====================================================================
def load_tokenizer_and_model():
    print("=== ÎNCARC TOKENIZER & MODEL MT5 ===")

    tokenizer = T5TokenizerFast(
        vocab_file=str(SP_MODEL_PATH),
        model_max_length=512,
    )

    tokenizer.pad_token = "<pad>"
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"

    tokenizer.save_pretrained(TOKENIZER_DIR)

    config = MT5Config.from_pretrained(BASE_MT5_NAME)
    config.vocab_size = len(tokenizer)
    config.decoder_start_token_id = tokenizer.pad_token_id

    model = MT5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))

    print(f"[INFO] Vocab_size tokenizer = {len(tokenizer)}")
    return model, tokenizer


# =====================================================================
# DATASET
# =====================================================================
class EmojiDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["ro"].astype(str).tolist()
        self.labels = df["emoji"].astype(str).tolist()
        self.tok = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        src = self.texts[i]
        tgt = self.labels[i]

        enc = self.tok(src, max_length=MAX_INPUT_LEN, truncation=True, padding="max_length", return_tensors="pt")
        dec = self.tok(tgt, max_length=MAX_TARGET_LEN, truncation=True, padding="max_length", return_tensors="pt")

        labels = dec["input_ids"].squeeze(0)
        labels[labels == self.tok.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


# =====================================================================
# TRAINING
# =====================================================================
def train_model(train_df, valid_df):

    model, tokenizer = load_tokenizer_and_model()

    train_ds = EmojiDataset(train_df, tokenizer)
    valid_ds = EmojiDataset(valid_df, tokenizer)

    collator = DataCollatorForSeq2Seq(tokenizer, model)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-4,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
        logging_dir=str(LOGS_DIR),
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    print("=== TRAINING MT5 ===")
    trainer.train()

    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    return trainer, tokenizer


# =====================================================================
# METRICS, PLOT, SAMPLES
# =====================================================================
def evaluate_model(trainer):
    res = trainer.evaluate()
    out = REPORTS_DIR / "metrics.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[OK] metrics.json → {out}")


def plot_loss(trainer):
    logs = trainer.state.log_history
    steps = [x["step"] for x in logs if "loss" in x]
    losses = [x["loss"] for x in logs if "loss" in x]

    if not steps:
        return

    plt.plot(steps, losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    out = PLOTS_DIR / "loss_curve.png"
    plt.savefig(out)
    plt.close()
    print(f"[OK] loss_curve.png → {out}")


def save_samples(trainer, tokenizer, valid_df):
    out = SAMPLES_DIR / "valid_samples.txt"
    lines = []

    model = trainer.model
    model.eval()

    for i in range(20):
        ro = valid_df.iloc[i]["ro"]
        gold = valid_df.iloc[i]["emoji"]

        inp = tokenizer(ro, return_tensors="pt", truncation=True).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inp, max_length=MAX_TARGET_LEN)
        pred = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        lines.append(f"[{i}] RO:   {ro}\n")
        lines.append(f"     GOLD: {gold}\n")
        lines.append(f"     PRED: {pred}\n\n")

    with out.open("w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[OK] Sample previews → {out}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    train_df, valid_df, test_df = load_csvs()
    build_corpus(train_df, valid_df, test_df)
    build_tokenizer()
    trainer, tokenizer = train_model(train_df, valid_df)
    evaluate_model(trainer)
    plot_loss(trainer)
    save_samples(trainer, tokenizer, valid_df)
    print("\n=== TOT PIPELINE-UL A RULAT CU SUCCES ===\n")


if __name__ == "__main__":
    main()
