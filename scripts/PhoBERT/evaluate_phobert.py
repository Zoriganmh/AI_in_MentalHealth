# -*- coding: utf-8 -*-
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__)) 
scripts_dir = os.path.dirname(current_dir)               
sys.path.insert(0, scripts_dir)                          

from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import evaluate
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    DATA_DIR, FEATURES_DIR, MODEL_DIR, OUTPUT_DIR, EVAL_DIR, NUM_LABELS, BATCH_SIZE
)

def load_model_and_tokenizer():
    model_dir = os.path.join(MODEL_DIR, "phobert_vi", "final")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=NUM_LABELS,
    )
    return model, tokenizer

def get_eval_args():
    use_cuda = torch.cuda.is_available()
    return TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, 'phobert_eval_tmp'),
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=use_cuda,
        report_to="none" 
    )

def get_compute_metrics():
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        results = {}
        results.update(acc_metric.compute(predictions=preds, references=labels))
        results.update(
            f1_metric.compute(
                predictions=preds, references=labels, average="macro"
            )
        )
        return results

    return compute_metrics

def save_metrics(metrics, split_name):
    phobert_eval_dir = os.path.join(EVAL_DIR, "phobert")
    os.makedirs(phobert_eval_dir, exist_ok=True)

    out_path = os.path.join(phobert_eval_dir, f"phobert_{split_name}_metrics.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Da luu ket qua do luong tap {split_name} vao {out_path}")

def save_detailed_evaluation(predictions, labels):
    phobert_eval_dir = os.path.join(EVAL_DIR, "phobert")
    os.makedirs(phobert_eval_dir, exist_ok=True)
    
    test_csv_path = os.path.join(DATA_DIR, "processed", "test_vi.csv")
    test_df = pd.read_csv(test_csv_path)
    
    if 'class_name' in test_df.columns and 'class_id' in test_df.columns:
        class_mapping = dict(zip(test_df['class_id'], test_df['class_name']))
        target_names = [class_mapping.get(i, f"Class_{i}") for i in range(NUM_LABELS)]
    else:
        target_names = [f"Class_{i}" for i in range(NUM_LABELS)]

    report = classification_report(labels, predictions, target_names=target_names, digits=4)
    report_path = os.path.join(phobert_eval_dir, "phobert_detailed_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report)
    print(f"Da luu bao cao chi tiet vao {report_path}")

    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="rocket_r", xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix - PhoBERT (Vietnamese)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    cm_path = os.path.join(phobert_eval_dir, "phobert_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Da luu Confusion Matrix vao {cm_path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on device: {device}")

    tokens_path = os.path.join(FEATURES_DIR, 'phobert_dataset_tokenized')
    dataset = load_from_disk(tokens_path)
    
    model, tokenizer = load_model_and_tokenizer()
    eval_args = get_eval_args()
    compute_metrics = get_compute_metrics()

    trainer = Trainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer, # <--- SỬA TẠI ĐÂY NỮA
        compute_metrics=compute_metrics,
    )

    for split in ["train", "validation", "test"]:
        print(f"\nEvaluating on {split.upper()}...")
        metrics = trainer.evaluate(eval_dataset=dataset[split])
        save_metrics(metrics, split)

    print("\nCreating detailed report and Confusion Matrix for test set...")
    predictions_output = trainer.predict(dataset["test"])
    preds = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids
    
    save_detailed_evaluation(preds, labels)
    print("\n=== EVALUATION COMPLETE ===")

if __name__ == "__main__":
    main()