# train_roberta.py
import sys
import os

# --- THÊM ĐOẠN NÀY LÊN TRÊN CÙNG ---
# Lấy đường dẫn hiện tại của file này
current_dir = os.path.dirname(os.path.abspath(__file__))
# Lùi ra 2 cấp (ví dụ: từ models/bert lùi ra models, lùi tiếp ra gốc)
root_dir = os.path.dirname(os.path.dirname(current_dir))
# Thêm thư mục gốc vào hệ thống để Python tìm thấy config.py
sys.path.append(root_dir)
# -----------------------------------
import torch
from datasets import load_from_disk
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    TrainingArguments,
    Trainer,
)
import evaluate

from config import (
    FEATURES_DIR,   # Đổi từ TOKENIZED_DIR sang FEATURES_DIR để tự động nối chuỗi
    MODEL_DIR,      # Đổi từ ROBERTA_MODEL_DIR sang MODEL_DIR
    LOGGING_DIR,
    NUM_LABELS,
    LEARNING_RATE,
    NUM_EPOCHS,
)

def load_tokenized_dataset(prefix):
    """Load dữ liệu đã tokenize tương ứng với từng prefix (both, posts, titles)"""
    tokens_path = os.path.join(FEATURES_DIR, f'roberta_{prefix}_tokenized')
    return load_from_disk(tokens_path)

def load_model_and_tokenizer():
    model_name = "roberta-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        hidden_dropout_prob=0.3, # Cấu hình dropout 0.3 theo bài báo
    )
    return model, tokenizer

def get_training_args(prefix, batch_size):
    """Khởi tạo arguments với batch_size và thư mục lưu trữ linh động"""
    use_cuda = torch.cuda.is_available()
    col_model_dir = os.path.join(MODEL_DIR, f"roberta_{prefix}")

    return TrainingArguments(
        output_dir=os.path.join(col_model_dir, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1", # Tối ưu hóa theo F1-score thay vì Accuracy
        logging_dir=os.path.join(LOGGING_DIR, f"roberta_{prefix}"),
        fp16=use_cuda,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

def get_compute_metrics():
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        results = {}
        results.update(acc_metric.compute(predictions=preds, references=labels))
        results.update(
            f1_metric.compute(
                predictions=preds, references=labels, average="macro"
            )
        )
        return results

    return compute_metrics

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Cấu hình 3 luồng: (tên_tiền_tố, batch_size tương ứng)
    configs = [
        ("both", 16),
        ("posts", 16),
        ("titles", 32)
    ]

    for prefix, batch_size in configs:
        print(f"\n{'='*50}")
        print(f"TRAINING RoBERTa ON: {prefix.upper()}")
        print(f"{'='*50}")

        print(f"Loading {prefix} tokenized dataset from disk...")
        tokenized_dataset = load_tokenized_dataset(prefix)

        print("Loading RoBERTa model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer()

        training_args = get_training_args(prefix, batch_size)
        compute_metrics = get_compute_metrics()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        print(f"Starting training for {prefix}...")
        trainer.train()

        # Lưu model final vào thư mục riêng biệt
        final_dir = os.path.join(MODEL_DIR, f"roberta_{prefix}", "final")
        os.makedirs(final_dir, exist_ok=True)

        print(f"Saving best model for {prefix} to {final_dir} ...")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

        print(f"Training complete for {prefix}. Model saved.\n")

if __name__ == "__main__":
    main()