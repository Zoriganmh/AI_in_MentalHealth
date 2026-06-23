# -*- coding: utf-8 -*-
import sys
import os
import torch
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

current_dir = os.path.dirname(os.path.abspath(__file__)) 
scripts_dir = os.path.dirname(current_dir)               
sys.path.insert(0, scripts_dir)                          

from config import (
    FEATURES_DIR, MODEL_DIR, NUM_LABELS, 
    LEARNING_RATE, NUM_EPOCHS, PHOBERT_MODEL_NAME, BATCH_SIZE
)

def load_phobert_dataset():
    tokens_path = os.path.join(FEATURES_DIR, 'phobert_dataset_tokenized')
    print(f"Loading tokenized dataset from {tokens_path}...")
    return load_from_disk(tokens_path)

def load_model_and_tokenizer():
    print("Loading PhoBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        PHOBERT_MODEL_NAME,
        num_labels=NUM_LABELS,
        hidden_dropout_prob=0.3,
        use_safetensors=True, 
    )
    return model, tokenizer

def get_training_args():
    use_cuda = torch.cuda.is_available()
    col_model_dir = os.path.join(MODEL_DIR, "phobert_vi")

    return TrainingArguments(
        output_dir=os.path.join(col_model_dir, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=use_cuda,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none" # Dọn dẹp cảnh báo logging_dir
    )

def get_compute_metrics():
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = torch.argmax(torch.tensor(logits), dim=-1)
        results = {}
        results.update(acc_metric.compute(predictions=preds, references=labels))
        results.update(f1_metric.compute(predictions=preds, references=labels, average="macro"))
        return results
        
    return compute_metrics

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    tokenized_dataset = load_phobert_dataset()
    model, tokenizer = load_model_and_tokenizer()
    training_args = get_training_args()
    compute_metrics = get_compute_metrics()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer, # <--- SỬA TẠI ĐÂY
        compute_metrics=compute_metrics,
    )

    print("\n=== STARTING TRAINING ===")
    trainer.train()

    final_dir = os.path.join(MODEL_DIR, "phobert_vi", "final")
    os.makedirs(final_dir, exist_ok=True)

    print(f"\nSaving best model to {final_dir} ...")
    trainer.save_model(final_dir)
    # Lưu lại tokenizer
    tokenizer.save_pretrained(final_dir)

    print("=== TRAINING COMPLETE ===")

if __name__ == "__main__":
    main()