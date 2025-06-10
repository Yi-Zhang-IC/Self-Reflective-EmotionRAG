from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import Trainer
import torch
from transformers import set_seed
from transformers import TrainerCallback
import json
from pathlib import Path
import pandas as pd

set_seed(11711)
# Load the CARER dataset
# The CARER dataset is a multi-label emotion classification dataset
# It contains 6 different emotions: joy, anger, sadness, fear, disgust, and surprise
# The dataset is available on Hugging Face Hub
dataset = load_dataset("dair-ai/emotion")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

class RealTimeLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            pass  # Create/clear file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_local_process_zero:
            logs = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
            logs["step"] = state.global_step
            logs["epoch"] = float(state.epoch) if state.epoch is not None else None

            # Save to JSONL file
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")

def preprocess(batch):
    tokenized = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

    one_hot_labels = []
    for label in batch["label"]:
        one_hot = [0.0] * 6  # <- Use float here
        one_hot[label] = 1.0
        one_hot_labels.append(one_hot)

    tokenized["labels"] = one_hot_labels
    return tokenized

encoded = dataset.map(preprocess, batched=True)
encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=False)

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=6,  # Change this depending on how you process the label set
    problem_type="multi_label_classification"
)

training_args = TrainingArguments(
    output_dir = "/root/emotion-retrieval-embeddings/outputs/carer_roberta",          # Where to save checkpoints and final model
    evaluation_strategy="steps",                    # Evaluate every N steps
    eval_steps=50,                                  # Evaluate every 50 steps
    save_strategy="steps",                          # Save model every N steps
    save_steps=50,                                  # Save every 50 steps
    report_to="none",                               # Disable reporting to WandB or other services
    logging_dir="/root/emotion-retrieval-embeddings/outputs/carer_roberta/logs",    # Directory for logs
    logging_steps=10,                               # Log training metrics every 10 steps
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    save_total_limit=1, 
    weight_decay=0.01,
    warmup_steps=500,
    seed=11711,
    load_best_model_at_end=True,                    # Reload best model according to `metric_for_best_model`
    metric_for_best_model="f1",                     # Use F1 to determine best model
    greater_is_better=True
)


def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    return {
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0)
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[RealTimeLoggerCallback("logs/carer_roberta_training.jsonl")]
)

trainer.train()