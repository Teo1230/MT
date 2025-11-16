import os
import json
import argparse
import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler  # <-- pentru AMP

MODEL_NAME = "dumitrescustefan/bert-base-romanian-cased-v1"


# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_emoji_vocab(*dfs: pd.DataFrame, col: str = "emoji") -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Construiește vocabular de emoji-character-level.
    Ia toate coloanele 'emoji' din train+valid+test, concatenează
    și colectează caracterele unice non-whitespace.
    """
    chars = set()
    for df in dfs:
        for s in df[col].astype(str):
            for ch in s:
                if not ch.isspace():
                    chars.add(ch)

    chars = sorted(chars)
    emoji2id = {ch: i for i, ch in enumerate(chars)}
    id2emoji = {i: ch for ch, i in emoji2id.items()}
    return emoji2id, id2emoji


def encode_multi_label(emoji_str: str, emoji2id: Dict[str, int]) -> np.ndarray:
    """
    Transformă un string de emoji într-un vector multi-label one-hot.
    Fiecare caracter unic din string devine 1 în vector.
    """
    vec = np.zeros(len(emoji2id), dtype=np.float32)
    for ch in set(str(emoji_str)):
        if ch in emoji2id:
            vec[emoji2id[ch]] = 1.0
    return vec


# ---------------------------
# Dataset
# ---------------------------

class Text2EmojiDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        emoji2id: Dict[str, int],
        max_length: int = 64,
        text_col: str = "ro",
        emoji_col: str = "emoji",
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.emoji2id = emoji2id
        self.max_length = max_length
        self.text_col = text_col
        self.emoji_col = emoji_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        emoji_str = str(row[self.emoji_col])

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = encode_multi_label(emoji_str, self.emoji2id)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


# ---------------------------
# Model
# ---------------------------

class EmojiBertClassifier(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # CLS token
        cls_rep = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_rep)
        return logits


# ---------------------------
# Train / Eval
# ---------------------------

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    loss_fn,
    use_amp: bool = False,
    scaler: GradScaler = None,
):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def eval_model(
    model,
    dataloader,
    device,
    loss_fn,
    threshold: float = 0.5,
):
    model.eval()
    total_loss = 0.0

    all_labels = []
    all_preds = []

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    # micro/macro F1 (multi-label)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return avg_loss, micro_f1, macro_f1


def save_plots(history: Dict, output_dir: str):
    # Loss plot
    plt.figure()
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss curve")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    # F1 plot
    plt.figure()
    plt.plot(history["val_micro_f1"], label="Val micro F1")
    plt.plot(history["val_macro_f1"], label="Val macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.title("F1 curve")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_curve.png"))
    plt.close()


# ---------------------------
# Inference helper
# ---------------------------

@torch.no_grad()
def predict_emojis(
    text: str,
    model,
    tokenizer,
    emoji2id: Dict[str, int],
    id2emoji: Dict[int, str],
    device,
    max_length: int = 64,
    threshold: float = 0.5,
) -> str:
    model.eval()
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.sigmoid(logits).squeeze(0)

    preds = (probs >= threshold).nonzero(as_tuple=True)[0].tolist()
    emojis = [id2emoji[i] for i in preds]
    return "".join(emojis)


# ---------------------------
# Experiment: bert_multilabel
# ---------------------------

def run_bert_multilabel(args):
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    print("=== Loading data ===")
    train_df = pd.read_csv(args.train_path)
    valid_df = pd.read_csv(args.valid_path)
    test_df = pd.read_csv(args.test_path)

    print("=== Building emoji vocabulary ===")
    emoji2id, id2emoji = build_emoji_vocab(train_df, valid_df, test_df, col=args.emoji_col)
    num_labels = len(emoji2id)
    print(f"Num emoji labels: {num_labels}")

    # Save vocab
    with open(os.path.join(args.output_dir, "emoji_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(emoji2id, f, ensure_ascii=False, indent=2)

    print("=== Loading tokenizer & model ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EmojiBertClassifier(num_labels=num_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    train_ds = Text2EmojiDataset(
        train_df,
        tokenizer=tokenizer,
        emoji2id=emoji2id,
        max_length=args.max_length,
        text_col=args.text_col,
        emoji_col=args.emoji_col,
    )
    valid_ds = Text2EmojiDataset(
        valid_df,
        tokenizer=tokenizer,
        emoji2id=emoji2id,
        max_length=args.max_length,
        text_col=args.text_col,
        emoji_col=args.emoji_col,
    )
    test_ds = Text2EmojiDataset(
        test_df,
        tokenizer=tokenizer,
        emoji2id=emoji2id,
        max_length=args.max_length,
        text_col=args.text_col,
        emoji_col=args.emoji_col,
    )

    # DataLoader mai optimizat
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 4 if device.type == "cuda" else 0,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_micro_f1": [],
        "val_macro_f1": [],
    }

    best_val_micro_f1 = 0.0
    best_epoch = -1
    best_model_path = os.path.join(args.output_dir, "best_model.pt")

    use_amp = (device.type == "cuda")
    scaler = GradScaler() if use_amp else None

    print("=== Training ===")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            loss_fn,
            use_amp=use_amp,
            scaler=scaler,
        )

        val_loss, val_micro_f1, val_macro_f1 = eval_model(
            model,
            valid_loader,
            device,
            loss_fn,
            threshold=args.threshold,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_micro_f1"].append(val_micro_f1)
        history["val_macro_f1"].append(val_macro_f1)

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val micro F1: {val_micro_f1:.4f} | "
            f"Val macro F1: {val_macro_f1:.4f}"
        )

        # Save best model
        if val_micro_f1 > best_val_micro_f1:
            best_val_micro_f1 = val_micro_f1
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "emoji2id": emoji2id,
                    "id2emoji": id2emoji,
                    "args": vars(args),
                },
                best_model_path,
            )
            print(f"--> Saved new best model at epoch {epoch} (micro F1={val_micro_f1:.4f})")

    # Save training history
    with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    save_plots(history, args.output_dir)

    print(f"\nBest epoch: {best_epoch} with Val micro F1 = {best_val_micro_f1:.4f}")

    # Load best model for final evaluation
    print("=== Loading best model for final evaluation (train/valid/test) ===")
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Ca să "vezi clar dacă a învățat" -> evaluăm pe train, valid, test cu best model
    train_loss_best, train_micro_f1_best, train_macro_f1_best = eval_model(
        model,
        train_loader,
        device,
        loss_fn,
        threshold=args.threshold,
    )

    val_loss_best, val_micro_f1_best, val_macro_f1_best = eval_model(
        model,
        valid_loader,
        device,
        loss_fn,
        threshold=args.threshold,
    )

    test_loss, test_micro_f1, test_macro_f1 = eval_model(
        model,
        test_loader,
        device,
        loss_fn,
        threshold=args.threshold,
    )

    metrics = {
        "best_epoch": best_epoch,
        "train_loss_best": train_loss_best,
        "train_micro_f1_best": train_micro_f1_best,
        "train_macro_f1_best": train_macro_f1_best,
        "val_loss_best": val_loss_best,
        "val_micro_f1_best": val_micro_f1_best,
        "val_macro_f1_best": val_macro_f1_best,
        "test_loss": test_loss,
        "test_micro_f1": test_micro_f1,
        "test_macro_f1": test_macro_f1,
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Final metrics (best model) ===")
    print(json.dumps(metrics, indent=2))


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment name (bert_multilabel)")
    parser.add_argument("--train_path", type=str, default="text2emoji_ro_train.csv")
    parser.add_argument("--valid_path", type=str, default="text2emoji_ro_valid.csv")
    parser.add_argument("--test_path", type=str, default="text2emoji_ro_test.csv")

    parser.add_argument("--text_col", type=str, default="ro")
    parser.add_argument("--emoji_col", type=str, default="emoji")

    parser.add_argument("--output_dir", type=str, default="outputs_bert_multilabel")

    # 20 epoci by default
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.exp == "bert_multilabel":
        run_bert_multilabel(args)
    else:
        raise ValueError(f"Experiment necunoscut: {args.exp}")


if __name__ == "__main__":
    main()
