import os
import json
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


MODEL_NAME = "dumitrescustefan/bert-base-romanian-cased-v1"


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_emoji_vocab(*dfs: pd.DataFrame, col: str = "emoji") -> Tuple[Dict[str, int], Dict[int, str]]:
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
    vec = np.zeros(len(emoji2id), dtype=np.float32)
    for ch in set(str(emoji_str)):
        if ch in emoji2id:
            vec[emoji2id[ch]] = 1.0
    return vec


# ============================================================
# Dataset
# ============================================================

class Text2EmojiDataset(Dataset):
    def __init__(self, df, tokenizer, emoji2id, max_length, text_col, emoji_col, augment):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.emoji2id = emoji2id
        self.max_length = max_length
        self.text_col = text_col
        self.emoji_col = emoji_col
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        emoji_str = str(row[self.emoji_col])

        if self.augment and random.random() > 0.5:
            text = text.lower()

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
            "labels": torch.tensor(labels),
        }


# ============================================================
# Model
# ============================================================

class EmojiBertClassifier(nn.Module):
    def __init__(self, num_labels, hidden_dropout=0.3, attention_dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)

        self.bert.config.attention_probs_dropout_prob = attention_dropout
        self.bert.config.hidden_dropout_prob = hidden_dropout

        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_rep)


# ============================================================
# Loss: Focal
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * focal_weight

        return (focal_weight * bce_loss).mean()


# ============================================================
# Training & Evaluation
# ============================================================

def train_one_epoch(model, loader, optimizer, scheduler, device, loss_fn, scaler, max_grad_norm):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            logits = model(ids, mask)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_model(model, loader, device, loss_fn, threshold=0.5):
    model.eval()
    total_loss = 0
    all_labels, all_preds, all_probs = [], [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(ids, mask)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)

    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return total_loss / len(loader), micro, macro, y_probs


# ============================================================
# Plots
# ============================================================

def save_plots(history, out_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Valid")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history["val_micro_f1"], label="Micro")
    plt.plot(history["val_macro_f1"], label="Macro")
    plt.legend()
    plt.title("F1 Curve")
    plt.savefig(os.path.join(out_dir, "f1_curve.png"))
    plt.close()


# ============================================================
# Threshold Search
# ============================================================

def find_threshold(y_true, y_probs):
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(y_true, (y_probs >= t).astype(float), average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ============================================================
# Decode Predictions
# ============================================================

def decode_predictions(probs, threshold, id2emoji):
    result = []
    for p in probs:
        idxs = np.where(p >= threshold)[0]
        result.append("".join(id2emoji[i] for i in idxs))
    return result


# ============================================================
# Compute pos_weight
# ============================================================

def compute_pos_weight(df, emoji2id, col, smoothing=1.0):
    labels = [encode_multi_label(s, emoji2id) for s in df[col].astype(str)]
    mat = np.stack(labels)

    pos = mat.sum(0) + smoothing
    neg = len(mat) - mat.sum(0) + smoothing

    pw = np.clip(neg / pos, 0.1, 10.0)
    return torch.tensor(pw, dtype=torch.float32)


# ============================================================
# MAIN TRAINING PROCESS
# ============================================================

def run_bert_multilabel(cfg):

    # -------------------------
    # Directory structure
    # -------------------------
    exp = cfg["exp"]
    logs_dir = os.path.join(exp, "logs")
    best_dir = os.path.join(exp, "best_model")
    preds_dir = os.path.join(exp, "predictions")
    plots_dir = os.path.join(exp, "plots")

    for d in [logs_dir, best_dir, preds_dir, plots_dir]:
        os.makedirs(d, exist_ok=True)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # -------------------------
    # Load Data
    # -------------------------
    train_df = pd.read_csv(cfg["train_path"])
    valid_df = pd.read_csv(cfg["valid_path"])
    test_df = pd.read_csv(cfg["test_path"])

    emoji2id, id2emoji = build_emoji_vocab(train_df, valid_df, test_df, col=cfg["emoji_col"])

    with open(os.path.join(best_dir, "emoji_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(emoji2id, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # datasets
    train_ds = Text2EmojiDataset(train_df, tokenizer, emoji2id, cfg["max_length"],
                                 cfg["text_col"], cfg["emoji_col"], augment=True)
    valid_ds = Text2EmojiDataset(valid_df, tokenizer, emoji2id, cfg["max_length"],
                                 cfg["text_col"], cfg["emoji_col"], augment=False)
    test_ds = Text2EmojiDataset(test_df, tokenizer, emoji2id, cfg["max_length"],
                                cfg["text_col"], cfg["emoji_col"], augment=False)

    # dataloaders
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"])

    # -------------------------
    # Model, Optimizer, Loss
    # -------------------------
    model = EmojiBertClassifier(
        num_labels=len(emoji2id),
        hidden_dropout=cfg["hidden_dropout"],
        attention_dropout=cfg["attention_dropout"]
    ).to(device)

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "bert" in n and not any(nd in n for nd in no_decay)],
             "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},

            {"params": [p for n, p in model.named_parameters() if "bert" in n and any(nd in n for nd in no_decay)],
             "lr": cfg["lr"], "weight_decay": 0.0},

            {"params": [p for n, p in model.named_parameters() if "classifier" in n],
             "lr": cfg["lr"] * cfg["classifier_lr_multiplier"], "weight_decay": cfg["weight_decay"]},
        ]
    )

    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    pos_weight = compute_pos_weight(train_df, emoji2id, cfg["emoji_col"], smoothing=cfg["pos_weight_smoothing"]).to(device)

    if cfg["loss_type"] == "focal":
        loss_fn = FocalLoss(cfg["focal_alpha"], cfg["focal_gamma"], pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # -------------------------
    # Training Loop
    # -------------------------
    history = {"train_loss": [], "val_loss": [], "val_micro_f1": [], "val_macro_f1": []}
    best_micro = 0
    patience_count = 0
    best_model_path = os.path.join(best_dir, "best_model.pt")

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['epochs']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device,
                                     loss_fn, scaler, cfg["max_grad_norm"])

        val_loss, micro, macro, val_probs = eval_model(model, valid_loader, device,
                                                       loss_fn, threshold=cfg["threshold"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_micro_f1"].append(micro)
        history["val_macro_f1"].append(macro)

        print(f"Loss {train_loss:.4f} | Val {val_loss:.4f} | micro {micro:.4f}")

        if micro > best_micro:
            best_micro = micro
            patience_count = 0
            torch.save({"model_state_dict": model.state_dict()}, best_model_path)
            print("Saved best model ✔")
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                print("Early stopping.")
                break

    # save training history
    with open(os.path.join(logs_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    save_plots(history, plots_dir)

    # ---------------------------------------------------------
    # Load best model
    # ---------------------------------------------------------
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # best threshold
    _, _, _, val_probs = eval_model(model, valid_loader, device, loss_fn, threshold=0.5)
    val_labels = np.concatenate([b["labels"].numpy() for b in valid_loader])
    best_t, best_val_f1 = find_threshold(val_labels, val_probs)

    print(f"Best threshold: {best_t} | F1={best_val_f1:.4f}")

    np.save(os.path.join(preds_dir, "val_probs.npy"), val_probs)
    np.save(os.path.join(preds_dir, "val_labels.npy"), val_labels)

    # ---------------------------------------------------------
    # Test evaluation
    # ---------------------------------------------------------
    test_loss_def, micro_def, macro_def, probs_def = eval_model(
        model, test_loader, device, loss_fn, threshold=cfg["threshold"]
    )
    test_loss_opt, micro_opt, macro_opt, probs_opt = eval_model(
        model, test_loader, device, loss_fn, threshold=best_t
    )

    metrics = {
        "best_val_micro_f1": float(best_micro),
        "optimal_threshold": float(best_t),

        "test_loss_default": float(test_loss_def),
        "test_micro_f1_default": float(micro_def),
        "test_macro_f1_default": float(macro_def),

        "test_loss_optimal": float(test_loss_opt),
        "test_micro_f1_optimal": float(micro_opt),
        "test_macro_f1_optimal": float(macro_opt),
    }

    with open(os.path.join(logs_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------
    np.save(os.path.join(preds_dir, "test_probs_default.npy"), probs_def)
    np.save(os.path.join(preds_dir, "test_probs_optimal.npy"), probs_opt)

    pred_def = decode_predictions(probs_def, cfg["threshold"], id2emoji)
    pred_opt = decode_predictions(probs_opt, best_t, id2emoji)

    pd.DataFrame({
        "text": test_df[cfg["text_col"]],
        "true": test_df[cfg["emoji_col"]],
        "pred_default": pred_def,
        "pred_optimal": pred_opt,
    }).to_csv(os.path.join(preds_dir, "test_predictions.csv"), index=False)

    print(f"Experiment {exp} finished ✔")


# ============================================================
# EXPERIMENT LIST (edit freely)
# ============================================================

EXPERIMENTS = [

    {
        "exp": "carla4.1",
        "train_path": "carla_train.csv",
        "valid_path": "carla_validate.csv",
        "test_path": "carla_test.csv",

        "text_col": "ro",
        "emoji_col": "emoji",

        "epochs": 100,
        "batch_size": 16,
        "lr": 2e-5,
        "classifier_lr_multiplier": 10,
        "weight_decay": 0.01,
        "max_length": 64,
        "warmup_ratio": 0.1,
        "threshold": 0.5,
        "max_grad_norm": 1.0,

        "hidden_dropout": 0.3,
        "attention_dropout": 0.1,
        "patience": 5,

        "loss_type": "focal",
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,

        "pos_weight_smoothing": 1.0,
        "seed": 42,
    },

    {
        "exp": "carla4.2",
        "train_path": "carla_train.csv",
        "valid_path": "carla_validate.csv",
        "test_path": "carla_test.csv",

        "text_col": "ro",
        "emoji_col": "emoji",

        "epochs": 100,
        "batch_size": 16,
        "lr": 1e-5,      # lower lr
        "classifier_lr_multiplier": 10,
        "weight_decay": 0.01,
        "max_length": 64,
        "warmup_ratio": 0.1,
        "threshold": 0.5,
        "max_grad_norm": 1.0,

        "hidden_dropout": 0.3,
        "attention_dropout": 0.1,
        "patience": 5,

        "loss_type": "bce",
        "pos_weight_smoothing": 1.0,
        "seed": 42,
    },
]


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

def run_experiments(experiments):
    for cfg in experiments:
        print("\n=========================================")
        print(f" Running experiment: {cfg['exp']} ")
        print("=========================================")
        run_bert_multilabel(cfg)


if __name__ == "__main__":
    run_experiments(EXPERIMENTS)
