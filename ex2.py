#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json
from typing import List
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import LoraConfig, get_peft_model


# -----------------------------
# ARGUMENTE CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Qwen2.5-0.5B text->emoji cu LoRA")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--validation_csv", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="ro")
    parser.add_argument("--emoji_column", type=str, default="emoji")

    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-0.5B-Instruct")

    parser.add_argument("--output_dir", type=str,
                        default="qwen_emoji")

    parser.add_argument("--num_train_epochs", type=float, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=5)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_eval_samples", type=int, default=50)

    return parser.parse_args()


# -----------------------------
# PREPROCESARE - INSTRUCT FORMAT
# -----------------------------
def preprocess_dataset(args, tokenizer):

    raw = load_dataset("csv", data_files={
        "train": args.train_csv,
        "validation": args.validation_csv
    })

    def preprocess(examples):
        texts = examples[args.text_column]
        emojis = examples[args.emoji_column]

        input_ids_list = []
        label_ids_list = []

        for t, e in zip(texts, emojis):

            prompt = f"<s>[INST] Scrie emoji care exprimă: \"{t}\" [/INST]"
            full_text = prompt + " " + e

            enc_full = tokenizer(
                full_text,
                max_length=args.max_length,
                truncation=True
            )

            enc_prompt = tokenizer(
                prompt,
                max_length=args.max_length,
                truncation=True,
                add_special_tokens=False
            )

            input_ids = enc_full["input_ids"]
            labels = input_ids.copy()

            prompt_len = len(enc_prompt["input_ids"])
            labels[:prompt_len] = [-100] * prompt_len

            input_ids_list.append(input_ids)
            label_ids_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "labels": label_ids_list,
        }

    train = raw["train"].map(
        preprocess,
        batched=True,
        remove_columns=raw["train"].column_names,
    )
    valid = raw["validation"].map(
        preprocess,
        batched=True,
        remove_columns=raw["validation"].column_names,
    )

    return train, valid, raw["validation"]


# -----------------------------
# LoRA
# -----------------------------
def apply_lora(model, args):

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# -----------------------------
# GENERARE SAMPLE-URI
# -----------------------------
def generate_samples(args, model, tokenizer, raw_valid):

    path = os.path.join(args.output_dir, "samples.jsonl")
    f = open(path, "w", encoding="utf-8")

    model.eval()

    n = min(args.num_eval_samples, len(raw_valid))

    for i in range(n):
        row = raw_valid[i]

        text = row[args.text_column]
        gold = row[args.emoji_column]

        prompt = f"<s>[INST] Scrie emoji care exprimă: \"{text}\" [/INST]"

        enc = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_length=args.max_length,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        dec = tokenizer.decode(out[0], skip_special_tokens=True)

        if dec.startswith(prompt):
            pred = dec[len(prompt):].strip()
        else:
            pred = dec.strip()

        f.write(json.dumps({
            "text": text,
            "emoji_gold": gold,
            "emoji_pred": pred
        }, ensure_ascii=False) + "\n")

    f.close()
    print(f"Saved samples → {path}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # LoRA
    model = apply_lora(model, args)

    # Dataset
    print("Preprocessing dataset...")
    train_ds, val_ds, raw_valid_ds = preprocess_dataset(args, tokenizer)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=["none"]
    )

    # Collator corect
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator
    )

    # Train
    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)

    # Metrics
    metrics = trainer.evaluate()
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Samples
    generate_samples(args, model, tokenizer, raw_valid_ds)

    print("FINISHED.")


if __name__ == "__main__":
    main()

