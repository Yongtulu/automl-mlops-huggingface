"""
SageMaker Training Job Script — Bio_ClinicalBERT Fine-Tuning
=============================================================
Runs inside a SageMaker HuggingFace Training Job container.

Expected environment variables (set by SageMaker):
    SM_CHANNEL_TRAIN  — path to training data
    SM_MODEL_DIR      — path to save the model

Usage (local test):
    python train.py --train-dir ./data --model-dir ./output --epochs 3
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir",  type=str,
                        default=os.environ.get("SM_CHANNEL_TRAIN", "./data"))
    parser.add_argument("--model-dir",  type=str,
                        default=os.environ.get("SM_MODEL_DIR", "./output"))
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len",    type=int, default=128)
    return parser.parse_args()


def load_and_encode(train_dir: str):
    df_train = pd.read_csv(f"{train_dir}/train_processed.csv")
    df_test  = pd.read_csv(f"{train_dir}/test_processed.csv")

    le = LabelEncoder()
    df_train["label"] = le.fit_transform(df_train["output_text"])
    df_test["label"]  = le.transform(df_test["output_text"])

    print(f"Train: {len(df_train)}, Test: {len(df_test)}, Classes: {len(le.classes_)}")
    return df_train, df_test, le


def tokenize_dataset(df_train, df_test, tokenizer, max_len: int):
    def tokenize(batch):
        return tokenizer(
            batch["input_text"],
            truncation=True,
            padding=True,
            max_length=max_len,
        )

    hf_train = Dataset.from_pandas(df_train[["input_text", "label"]])
    hf_test  = Dataset.from_pandas(df_test[["input_text", "label"]])
    hf_train = hf_train.map(tokenize, batched=True)
    hf_test  = hf_test.map(tokenize, batched=True)
    return hf_train, hf_test


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def main():
    args = parse_args()
    print(f"Args: {args}")

    # ── Data ──────────────────────────────────────────────────────────────
    df_train, df_test, le = load_and_encode(args.train_dir)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_train, hf_test = tokenize_dataset(
        df_train, df_test, tokenizer, args.max_len
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(le.classes_)
    )

    # ── Training ──────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",       # new name (was evaluation_strategy)
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_test,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ── Evaluate ──────────────────────────────────────────────────────────
    results = trainer.evaluate()
    print(f"\nFinal Accuracy: {results['eval_accuracy']:.4f}")

    preds = np.argmax(trainer.predict(hf_test).predictions, axis=-1)
    print("\nClassification Report:")
    print(classification_report(df_test["label"], preds,
                                target_names=le.classes_))

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    with open(f"{args.model_dir}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print(f"\nModel saved to {args.model_dir}")


if __name__ == "__main__":
    main()
