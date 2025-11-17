import os
import json
import random
import logging
import copy
from datetime import datetime
from typing import Dict, Tuple
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import f1_score, precision_score, recall_score

import matplotlib.pyplot as plt
from tqdm import tqdm


# Base directory for all outputs
BASE_DIR = "ex5"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(BASE_DIR, f"run_{TIMESTAMP}")

# Model configurations to try
MODEL_CONFIGS = [
    {
        "name": "bert-base-romanian",
        "model_name": "dumitrescustefan/bert-base-romanian-cased-v1",
        "type": "bert",
    },
    {
        "name": "distilbert-base-multilingual",
        "model_name": "distilbert-base-multilingual-cased",
        "type": "distilbert",
    },
    {
        "name": "xlm-roberta-base",
        "model_name": "xlm-roberta-base",
        "type": "roberta",
    },
]


def setup_logging(log_dir):
    """Setup logging to both file and console"""
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "training.log")

    # Create logger
    logger = logging.getLogger("emoji_training")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_emoji_vocab(
    *dfs: pd.DataFrame, col: str = "emoji"
) -> Tuple[Dict[str, int], Dict[int, str]]:
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


class Text2EmojiDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        emoji2id,
        max_length=64,
        text_col="ro",
        emoji_col="emoji",
        augment=False,
    ):
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
            "labels": torch.tensor(labels, dtype=torch.float),
        }


class EmojiBertClassifier(nn.Module):
    def __init__(
        self, model_name, num_labels, hidden_dropout=0.3, attention_dropout=0.1
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        # Apply dropout
        if hasattr(self.bert.config, "attention_probs_dropout_prob"):
            self.bert.config.attention_probs_dropout_prob = attention_dropout
        if hasattr(self.bert.config, "hidden_dropout_prob"):
            self.bert.config.hidden_dropout_prob = hidden_dropout

        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

        # optional: freeze some layers if you want
        # for param in self.bert.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_rep)


class EmojiDistilBertClassifier(nn.Module):
    def __init__(self, model_name, num_labels, hidden_dropout=0.3):
        super().__init__()
        self.distilbert = AutoModel.from_pretrained(model_name)

        if hasattr(self.distilbert.config, "dropout"):
            self.distilbert.config.dropout = hidden_dropout

        hidden_size = self.distilbert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_rep = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_rep)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5

        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha >= 0:
            alpha_t = torch.where(targets >= 0.5, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        return (focal_weight * bce_loss).mean()


def train_one_epoch(
    model, dataloader, optimizer, scheduler, device, loss_fn, scaler, max_grad_norm=1.0
):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def eval_model(model, dataloader, device, loss_fn, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

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
        all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_probs = np.concatenate(all_probs, axis=0)

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="micro", zero_division=0)

    return avg_loss, micro_f1, macro_f1, precision, recall, y_probs, y_true


def find_optimal_threshold(y_true, y_probs, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    best_f1 = 0.0
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(float)
        f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, best_f1


def compute_pos_weight(train_df, emoji2id, emoji_col="emoji", smoothing=1.0):
    labels_list = []
    for s in train_df[emoji_col].astype(str):
        labels_list.append(encode_multi_label(s, emoji2id))
    label_matrix = np.stack(labels_list, axis=0)

    pos_counts = label_matrix.sum(axis=0) + smoothing
    N = label_matrix.shape[0]
    neg_counts = N - label_matrix.sum(axis=0) + smoothing
    pos_weight = np.clip(neg_counts / pos_counts, 0.1, 10.0)

    return torch.tensor(pos_weight, dtype=torch.float32)


def generate_hyperparameter_configs():
    """Generate diverse hyperparameter configurations"""
    configs = []

    # Define search space - optimized ranges
    lr_values = [1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5]
    dropout_values = [0.2, 0.3, 0.4, 0.5]
    weight_decay_values = [0.01, 0.03, 0.05, 0.1]
    batch_sizes = [16, 32]
    label_smoothing_values = [0.0, 0.05, 0.1, 0.15]
    focal_gamma_values = [1.5, 2.0, 2.5]
    scheduler_types = ["linear", "cosine"]

    # Create diverse configs
    for lr, dropout, wd, bs, ls, gamma, sched in product(
        lr_values[:3],
        dropout_values[:2],
        weight_decay_values[:2],
        batch_sizes,
        label_smoothing_values[:2],
        focal_gamma_values[:2],
        scheduler_types,
    ):
        configs.append(
            {
                "lr": lr,
                "hidden_dropout": dropout,
                "attention_dropout": dropout * 0.5,
                "weight_decay": wd,
                "batch_size": bs,
                "label_smoothing": ls,
                "focal_gamma": gamma,
                "focal_alpha": 0.25,
                "classifier_lr_multiplier": 10.0,
                "scheduler_type": sched,
            }
        )

    # Add random configs
    while len(configs) < 50:
        configs.append(
            {
                "lr": random.choice(lr_values),
                "hidden_dropout": random.choice(dropout_values),
                "attention_dropout": random.choice(dropout_values) * 0.5,
                "weight_decay": random.choice(weight_decay_values),
                "batch_size": random.choice(batch_sizes),
                "label_smoothing": random.choice(label_smoothing_values),
                "focal_gamma": random.choice(focal_gamma_values),
                "focal_alpha": 0.25,
                "classifier_lr_multiplier": 10.0,
                "scheduler_type": random.choice(scheduler_types),
            }
        )

    return configs[:50]


def make_dataloaders(
    train_df, valid_df, tokenizer, emoji2id, batch_size, device, augment_train=True
):
    """Utility to create dataloaders with sensible defaults per device."""
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    train_ds = Text2EmojiDataset(
        train_df, tokenizer, emoji2id, augment=augment_train
    )
    valid_ds = Text2EmojiDataset(valid_df, tokenizer, emoji2id, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, valid_loader


def train_single_config(
    config,
    model_config,
    train_df,
    valid_df,
    emoji2id,
    id2emoji,
    device,
    run_id,
    logger,
):
    """Train a single configuration"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Run {run_id}/50 | Model: {model_config['name']}")
    logger.info(
        "Config: "
        f"lr={config['lr']:.0e}, dropout={config['hidden_dropout']}, "
        f"wd={config['weight_decay']}, bs={config['batch_size']}, "
        f"ls={config['label_smoothing']}, gamma={config['focal_gamma']}, "
        f"scheduler={config['scheduler_type']}"
    )
    logger.info(f"{'='*70}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
        train_loader, valid_loader = make_dataloaders(
            train_df,
            valid_df,
            tokenizer,
            emoji2id,
            batch_size=config["batch_size"],
            device=device,
            augment_train=True,
        )

        # Create model based on type
        if model_config["type"] == "distilbert":
            model = EmojiDistilBertClassifier(
                model_name=model_config["model_name"],
                num_labels=len(emoji2id),
                hidden_dropout=config["hidden_dropout"],
            ).to(device)
        else:
            model = EmojiBertClassifier(
                model_name=model_config["model_name"],
                num_labels=len(emoji2id),
                hidden_dropout=config["hidden_dropout"],
                attention_dropout=config["attention_dropout"],
            ).to(device)

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (("bert" in n) or ("distilbert" in n) or ("roberta" in n))
                    and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": config["weight_decay"],
                "lr": config["lr"],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (("bert" in n) or ("distilbert" in n) or ("roberta" in n))
                    and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": config["lr"],
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if "classifier" in n
                ],
                "weight_decay": config["weight_decay"],
                "lr": config["lr"] * config["classifier_lr_multiplier"],
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-8
        )

        # Scheduler
        num_training_steps = len(train_loader) * 20
        num_warmup_steps = int(0.1 * num_training_steps)

        if config["scheduler_type"] == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps
            )

        # Loss
        pos_weight = compute_pos_weight(train_df, emoji2id).to(device)
        loss_fn = FocalLoss(
            alpha=config["focal_alpha"],
            gamma=config["focal_gamma"],
            pos_weight=pos_weight,
            label_smoothing=config["label_smoothing"],
        )

        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        best_val_f1 = 0.0
        best_epoch = -1
        patience_counter = 0
        patience = 5

        for epoch in range(1, 21):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                loss_fn,
                scaler,
            )
            (
                val_loss,
                val_micro_f1,
                val_macro_f1,
                val_prec,
                val_rec,
                _,
                _,
            ) = eval_model(model, valid_loader, device, loss_fn)

            logger.info(
                f"Epoch {epoch:2d}: Train={train_loss:.4f} | Val={val_loss:.4f} | "
                f"F1={val_micro_f1:.4f} | P={val_prec:.4f} | R={val_rec:.4f}"
            )

            if val_micro_f1 > best_val_f1:
                best_val_f1 = val_micro_f1
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stop at epoch {epoch}")
                    break

        logger.info(f"Best: epoch {best_epoch}, Val F1 = {best_val_f1:.4f}")

        return {
            "config": config,
            "model_config": model_config,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "run_id": run_id,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Run {run_id} failed with error: {str(e)}")
        return {
            "config": config,
            "model_config": model_config,
            "best_val_f1": 0.0,
            "best_epoch": -1,
            "run_id": run_id,
            "success": False,
            "error": str(e),
        }


@torch.no_grad()
def generate_predictions(model, dataloader, id2emoji, device, threshold, df_original):
    """Generate predictions CSV"""
    model.eval()
    all_preds = []
    all_probs_list = []

    for batch in tqdm(dataloader, desc="Predicting", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits)

        for prob_vec in probs:
            indices = (prob_vec >= threshold).nonzero(as_tuple=True)[0].tolist()
            emojis = "".join([id2emoji[i] for i in indices])
            all_preds.append(emojis)

            # Store top probabilities for analysis
            top_probs, top_indices = torch.topk(
                prob_vec, min(5, len(prob_vec))
            )
            prob_info = {
                id2emoji[idx.item()]: prob.item()
                for idx, prob in zip(top_indices, top_probs)
            }
            all_probs_list.append(json.dumps(prob_info, ensure_ascii=False))

    df_output = df_original.copy()
    df_output["predicted_emoji"] = all_preds
    df_output["top_5_probs"] = all_probs_list
    return df_output


def save_plots(results, output_dir):
    """Save comparison plots"""
    successful = [r for r in results if r["success"]]
    if not successful:
        return

    # F1 score comparison
    f1_scores = [r["best_val_f1"] for r in successful]

    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(f1_scores)), f1_scores, alpha=0.6, s=100)
    plt.xlabel("Configuration", fontsize=12)
    plt.ylabel("Validation F1 Score", fontsize=12)
    plt.title("Hyperparameter Tuning Results", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tuning_scatter.png"), dpi=150)
    plt.close()

    # Top 10 configurations
    sorted_results = sorted(
        successful, key=lambda x: x["best_val_f1"], reverse=True
    )[:10]
    names = [
        f"{r['model_config']['name'][:20]}\n(run {r['run_id']})"
        for r in sorted_results
    ]
    scores = [r["best_val_f1"] for r in sorted_results]

    plt.figure(figsize=(14, 8))
    bars = plt.barh(names, scores)
    plt.xlabel("Validation F1 Score", fontsize=12)
    plt.title("Top 10 Configurations", fontsize=14)
    plt.xlim(0, max(scores) * 1.1)

    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(i / len(bars)))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_10_configs.png"), dpi=150)
    plt.close()


def main():
    set_seed(42)

    # Setup directories and logging
    os.makedirs(RUN_DIR, exist_ok=True)
    logger = setup_logging(RUN_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Output directory: {RUN_DIR}")

    # Load data
    logger.info("\n=== Loading data ===")
    train_df = pd.read_csv("text2emoji_ro_train.csv")
    valid_df = pd.read_csv("text2emoji_ro_valid.csv")
    test_df = pd.read_csv("text2emoji_ro_test.csv")
    logger.info(
        f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}"
    )

    # Build vocab
    logger.info("=== Building vocab ===")
    emoji2id, id2emoji = build_emoji_vocab(train_df, valid_df, test_df)
    logger.info(f"Emoji vocab size: {len(emoji2id)}")

    with open(
        os.path.join(RUN_DIR, "emoji_vocab.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(emoji2id, f, ensure_ascii=False, indent=2)

    # Generate configs
    logger.info("\n=== Generating 50 hyperparameter configurations ===")
    configs = generate_hyperparameter_configs()
    logger.info(f"Generated {len(configs)} configurations")

    # Train all model architectures with all configs
    all_results = []
    run_counter = 1

    for model_config in MODEL_CONFIGS:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# TESTING MODEL: {model_config['name']}")
        logger.info(f"# Model path: {model_config['model_name']}")
        logger.info(f"{'#'*70}\n")

        for config in configs:
            result = train_single_config(
                config,
                model_config,
                train_df,
                valid_df,
                emoji2id,
                id2emoji,
                device,
                run_counter,
                logger,
            )
            all_results.append(result)
            run_counter += 1

            # Save intermediate results
            with open(
                os.path.join(RUN_DIR, "all_tuning_results.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(all_results, f, indent=2, default=str)

            # Save progress summary
            successful = [r for r in all_results if r["success"]]
            if successful:
                best_so_far = max(
                    successful, key=lambda x: x["best_val_f1"]
                )
                logger.info(f"\n>>> Progress: {len(all_results)} runs completed")
                logger.info(
                    f">>> Best so far: {best_so_far['best_val_f1']:.4f} "
                    f"(Run {best_so_far['run_id']}, {best_so_far['model_config']['name']})\n"
                )

    # Save plots for all tuning runs
    save_plots(all_results, RUN_DIR)

    # Find overall best config
    successful_results = [r for r in all_results if r["success"]]
    if not successful_results:
        logger.error("No successful runs!")
        return

    best_result = max(successful_results, key=lambda x: x["best_val_f1"])

    logger.info(f"\n{'='*70}")
    logger.info("BEST CONFIGURATION FOUND!")
    logger.info(f"{'='*70}")
    logger.info(f"Run: {best_result['run_id']}")
    logger.info(f"Model: {best_result['model_config']['name']}")
    logger.info(f"Validation F1: {best_result['best_val_f1']:.4f}")
    logger.info(f"Config: {json.dumps(best_result['config'], indent=2)}")
    logger.info(f"{'='*70}\n")

    # Train final model with best config
    logger.info("=== Training final model with best configuration ===")
    best_config = best_result["config"]
    best_model_config = best_result["model_config"]

    tokenizer = AutoTokenizer.from_pretrained(best_model_config["model_name"])

    # Dataloaders
    train_loader, valid_loader = make_dataloaders(
        train_df,
        valid_df,
        tokenizer,
        emoji2id,
        batch_size=best_config["batch_size"],
        device=device,
        augment_train=True,
    )

    # Test loader
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0
    test_ds = Text2EmojiDataset(test_df, tokenizer, emoji2id, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=best_config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # Create best model
    if best_model_config["type"] == "distilbert":
        model = EmojiDistilBertClassifier(
            model_name=best_model_config["model_name"],
            num_labels=len(emoji2id),
            hidden_dropout=best_config["hidden_dropout"],
        ).to(device)
    else:
        model = EmojiBertClassifier(
            model_name=best_model_config["model_name"],
            num_labels=len(emoji2id),
            hidden_dropout=best_config["hidden_dropout"],
            attention_dropout=best_config["attention_dropout"],
        ).to(device)

    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    # Optimizer for final training
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (("bert" in n) or ("distilbert" in n) or ("roberta" in n))
                and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": best_config["weight_decay"],
            "lr": best_config["lr"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (("bert" in n) or ("distilbert" in n) or ("roberta" in n))
                and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": best_config["lr"],
        },
        {
            "params": [
                p for n, p in model.named_parameters() if "classifier" in n
            ],
            "weight_decay": best_config["weight_decay"],
            "lr": best_config["lr"]
            * best_config["classifier_lr_multiplier"],
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-8
    )

    # Scheduler
    num_training_steps = len(train_loader) * 20
    num_warmup_steps = int(0.1 * num_training_steps)
    if best_config["scheduler_type"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

    # Loss
    pos_weight = compute_pos_weight(train_df, emoji2id).to(device)
    loss_fn = FocalLoss(
        alpha=best_config["focal_alpha"],
        gamma=best_config["focal_gamma"],
        pos_weight=pos_weight,
        label_smoothing=best_config["label_smoothing"],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_f1 = 0.0
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    patience = 5

    for epoch in range(1, 21):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            loss_fn,
            scaler,
        )
        (
            val_loss,
            val_micro_f1,
            val_macro_f1,
            val_prec,
            val_rec,
            val_probs,
            val_labels,
        ) = eval_model(model, valid_loader, device, loss_fn)

        logger.info(
            f"[FINAL] Epoch {epoch:2d}: Train={train_loss:.4f} | Val={val_loss:.4f} | "
            f"F1={val_micro_f1:.4f} | P={val_prec:.4f} | R={val_rec:.4f}"
        )

        if val_micro_f1 > best_val_f1:
            best_val_f1 = val_micro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[FINAL] Early stop at epoch {epoch}")
                break

    logger.info(f"[FINAL] Best epoch {best_epoch}, Val micro-F1 = {best_val_f1:.4f}")

    # Load best weights
    model.load_state_dict(best_state)

    # Find optimal threshold on validation set
    best_thresh, best_thresh_f1 = find_optimal_threshold(val_labels, val_probs)
    logger.info(
        f"[FINAL] Best threshold on validation: {best_thresh:.3f} "
        f"(F1={best_thresh_f1:.4f})"
    )

    # Evaluate on test set
    (
        test_loss,
        test_micro_f1,
        test_macro_f1,
        test_prec,
        test_rec,
        _,
        _,
    ) = eval_model(model, test_loader, device, loss_fn, threshold=best_thresh)
    logger.info(
        f"[TEST] Loss={test_loss:.4f} | micro-F1={test_micro_f1:.4f} | "
        f"macro-F1={test_macro_f1:.4f} | P={test_prec:.4f} | R={test_rec:.4f}"
    )

    # Generate predictions CSV for test set
    test_predictions_df = generate_predictions(
        model,
        test_loader,
        id2emoji,
        device,
        threshold=best_thresh,
        df_original=test_df,
    )
    pred_path = os.path.join(RUN_DIR, "text2emoji_ro_test_predictions.csv")
    test_predictions_df.to_csv(pred_path, index=False, encoding="utf-8")
    logger.info(f"Saved test predictions to: {pred_path}")


if __name__ == "__main__":
    main()
