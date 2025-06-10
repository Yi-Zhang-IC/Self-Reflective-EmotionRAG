from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)
import torch
from torch.utils.data import Dataset
from typing import Union
import json
from transformers import TrainerCallback
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np
import os
import shutil


# Download a fresh, working copy of the tokenizer and overwrite the broken one
AutoTokenizer.from_pretrained("roberta-base").save_pretrained("models/roberta-base-go_emotions")

@torch.no_grad()
def evaluate_embeddings_at_ks(
    model,
    dataset,
    device,
    batch_size=128,
    ks=[1, 5, 10, 20],
    num_classes=28
):
    from collections import defaultdict
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    model.eval()

    # Step 1: Embed the full dataset
    all_embeddings = []
    all_labels = []

    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Get encoder outputs (skip classification head)
        encoder = model.base_model
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        embeddings = F.normalize(cls_embeddings, p=2, dim=1)

        all_embeddings.append(embeddings)
        all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]
    all_labels = torch.cat(all_labels, dim=0).to(device)  # [N, C]
    N = all_labels.size(0)

    # Step 2: Precompute cosine similarity matrix
    sim_matrix = torch.matmul(all_embeddings, all_embeddings.T)
    sim_matrix.fill_diagonal_(-float('inf'))

    # Step 3: Evaluate at each k
    results = {}

    for k in ks:
        topk_scores, topk_indices = torch.topk(sim_matrix, k=k, dim=1)

        true_positives = 0
        total_relevant = 0
        total_predicted = N * k

        ap_list = []
        ndcg_list = []
        per_class_precisions = {c: [] for c in range(num_classes)}

        for i in range(N):
            query_label = all_labels[i]
            retrieved_labels = all_labels[topk_indices[i]]  # [k, num_classes]
            relevance = (retrieved_labels * query_label).sum(dim=1) > 0  # [k]

            tp = relevance.sum().item()
            true_positives += tp
            total_relevant += query_label.sum().item()

            precision_i = tp / k
            for c in torch.nonzero(query_label, as_tuple=False).squeeze(1).tolist():
                per_class_precisions[c].append(precision_i)

            # AP
            hits, ap = 0, 0.0
            for rank, rel in enumerate(relevance, 1):
                if rel:
                    hits += 1
                    ap += hits / rank
            ap /= hits if hits > 0 else 1.0
            ap_list.append(ap)

            # nDCG
            gains = relevance.float()
            discounts = torch.log2(torch.arange(2, k + 2, device=device).float())
            dcg = (gains / discounts).sum()
            ideal_gains = torch.sort(gains, descending=True).values
            ideal_dcg = (ideal_gains / discounts).sum()
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcg_list.append(ndcg)

        class_avg_precisions = [
            sum(per_class_precisions[c]) / len(per_class_precisions[c]) if per_class_precisions[c] else 0.0
            for c in range(num_classes)
        ]
        macro_precision = sum(class_avg_precisions) / num_classes

        precision = true_positives / total_predicted
        recall = true_positives / total_relevant if total_relevant > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        mAP = sum(ap_list) / len(ap_list)
        mean_nDCG = sum(ndcg_list) / len(ndcg_list)

        results[f"@{k}"] = {
            "macro_precision": macro_precision,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mAP": mAP,
            "nDCG": mean_nDCG
        }

    return results

class EmotionDataset(Dataset):
    def __init__(self, path: Union[Path, str], tokenizer, num_classes: int = 28, max_length: int = 256):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes
        self.labels = [s["labels"] for s in self.samples]

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        label = item["labels"]
        label_vector = torch.zeros(self.num_classes)
        for l in label:
            label_vector[l] = 1

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": label_vector
        }

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return self.labels

class RealTimeLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_local_process_zero:
            # Only keep loss-related logs
            filtered_logs = {
                k: float(v) for k, v in logs.items()
                if isinstance(v, (int, float)) and k in {"loss", "grad_norm"}
            }
            if not filtered_logs:
                return

            filtered_logs["step"] = state.global_step
            filtered_logs["epoch"] = float(state.epoch) if state.epoch is not None else None
            with open(self.log_path, "a") as f:
                f.write(json.dumps(filtered_logs) + "\n")

class MultiMetricSaverCallback(TrainerCallback):
    def __init__(self, base_dir, monitor_metrics):
        self.best_scores = {metric: -float("inf") for metric in monitor_metrics}
        self.monitor_metrics = monitor_metrics
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.base_dir / "metric_eval.jsonl"
        with open(self.log_path, "w") as f:
            pass

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None and hasattr(self, "trainer"):
            trainer = self.trainer
        if trainer is None:
            print("[MetricSaver] Cannot save: trainer is None.")
            return

        model = trainer.model
        tokenizer = trainer.tokenizer

        for metric in self.monitor_metrics:
            score = metrics.get(metric, None)
            if score is not None and score > self.best_scores[metric]:
                self.best_scores[metric] = score
                ckpt_path = self.base_dir / f"best_{metric}_ckpt"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                print(f"[MetricSaver] Saved new best model for {metric}: {score:.4f}")

        # ✅ Log all evaluation metrics to file
        log_record = {
            "step": state.global_step,
            "epoch": float(state.epoch) if state.epoch else None
        }
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                log_record[k] = float(v)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_record) + "\n")

class EmbeddingRetrievalMonitorCallback(TrainerCallback):
    def __init__(self, base_dir, dataset, ks=[1, 3, 6]):
        self.best_avg_retrieval = -float("inf")
        self.base_dir = Path(base_dir)
        self.ks = ks
        self.dataset = dataset  # should be validation set
        self.ckpt_dir = self.base_dir / "best_embedding_retrieval_ckpt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None and hasattr(self, "trainer"):
            trainer = self.trainer
        if trainer is None:
            return

        model = trainer.model.to(args.device)
        tokenizer = trainer.tokenizer

        results = evaluate_embeddings_at_ks(
            model=model,
            dataset=self.dataset,
            device=args.device,
            ks=self.ks,
            num_classes=model.config.num_labels
        )

        avg_acc = np.mean([results[f"@{k}"]["precision"] for k in self.ks])

        if avg_acc > self.best_avg_retrieval:
            self.best_avg_retrieval = avg_acc
            model.save_pretrained(self.ckpt_dir)
            tokenizer.save_pretrained(self.ckpt_dir)
            with open(self.ckpt_dir / "retrieval_score.json", "w") as f:
                json.dump({"avg_topk_accuracy": float(avg_acc), "step": state.global_step}, f, indent=2)

        # Log to JSONL
        log_path = self.base_dir / "embedding_eval.jsonl"
        log_record = {
            "step": state.global_step,
            "epoch": float(state.epoch) if state.epoch else None,
            "avg_retrieval_precision": float(avg_acc),
        }
        for k in self.ks:
            try:
                log_record[f"precision@{k}"] = float(results[f"@{k}"]["precision"])
                log_record[f"f1@{k}"] = float(results[f"@{k}"]["f1"])
                log_record[f"mAP@{k}"] = float(results[f"@{k}"]["mAP"])
                log_record[f"nDCG@{k}"] = float(results[f"@{k}"]["nDCG"])
            except Exception:
                pass

        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(log_record) + "\n")
                f.flush()
        except Exception:
            pass

def compute_topk_accuracy(logits, labels, ks=[1, 3, 6]):
    topk_acc = {}
    sorted_indices = torch.tensor(logits).sigmoid().numpy().argsort(axis=1)[:, ::-1]
    labels = labels.numpy()

    for k in ks:
        hits = []
        for i in range(len(labels)):
            topk_preds = sorted_indices[i, :k]
            gold = set(np.where(labels[i] == 1)[0])
            hits.append(len(gold.intersection(topk_preds)) > 0)
        topk_acc[f"top{k}_acc"] = np.mean(hits)

    # Also return the average over these
    topk_acc["topk_mean_acc"] = np.mean([topk_acc[f"top{k}_acc"] for k in ks])
    return topk_acc


def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    labels = torch.tensor(labels)

    results = {
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0)
    }

    results.update(compute_topk_accuracy(logits, labels, ks=[1, 3, 6]))
    return results

class EpochCheckpointSaverCallback(TrainerCallback):
    def __init__(self, save_dir, save_epochs):
        """
        Args:
            save_dir (Path): Path to the trainer's output_dir
            save_epochs (List[int]): Epochs at which to archive the checkpoint
        """
        self.save_dir = Path(save_dir)
        self.save_epochs = set(save_epochs)
        self.saved_epochs = set()

    def _copy_latest_checkpoint(self, dest_name):
        """Find the latest step checkpoint and copy it to a new named checkpoint."""
        ckpts = list(self.save_dir.glob("checkpoint-*"))
        if not ckpts:
            print(f"[EpochSaver] No checkpoint found to archive as {dest_name}")
            return

        # Sort by modification time to get the latest
        latest_ckpt = max(ckpts, key=os.path.getmtime)
        dest_ckpt = self.save_dir / dest_name

        if dest_ckpt.exists():
            print(f"[EpochSaver] Destination {dest_ckpt} already exists. Skipping.")
            return

        print(f"[EpochSaver] Copying {latest_ckpt} to {dest_ckpt}")
        shutil.copytree(latest_ckpt, dest_ckpt)

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        if current_epoch in self.save_epochs and current_epoch not in self.saved_epochs:
            self._copy_latest_checkpoint(f"checkpoint-epoch-{current_epoch}")
            self.saved_epochs.add(current_epoch)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._copy_latest_checkpoint("checkpoint-final")
        return control
    
# ===============================
# Setup paths and configuration
# ===============================
set_seed(11711)
base_path = Path.cwd()
model_checkpoint_path = base_path / "models" / "roberta-base-go_emotions" 
tokenizer_path = base_path / "models" / "roberta-base-go_emotions"

# Dataset paths
data_dir = base_path / "data" / "augmented_go_emotion"
train_path = data_dir / "train.jsonl"
val_path = data_dir / "validation.jsonl"
test_path = data_dir / "test.jsonl"

# ===============================
# Load tokenizer and datasets
# ===============================
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
train_dataset = EmotionDataset(train_path, tokenizer)
val_dataset = EmotionDataset(val_path, tokenizer)
test_dataset = EmotionDataset(test_path, tokenizer)

# ===============================
# Load the model (CARER-pretrained)
# ===============================
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint_path,
    num_labels=28,
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True
)

# ===============================
# Training arguments
# ===============================
training_args = TrainingArguments(
    output_dir=str(base_path / "outputs" / "goemotions_notransfer"),
    evaluation_strategy="steps",       # <-- must match
    save_strategy="steps",             # <-- must match
    eval_steps=50,
    save_steps=50,
    save_total_limit=5,
    logging_dir=str(base_path / "outputs" / "goemotions_notransfer" / "logs"),
    logging_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    seed=11711,
    lr_scheduler_type="constant",
    load_best_model_at_end=True,       # <-- now valid
    metric_for_best_model="f1",
    greater_is_better=True
)

# ===============================
# Callbacks for metrics and model saving
# ===============================
multi_metric_callback = MultiMetricSaverCallback(
    base_dir=base_path / "outputs" / "goemotions_notransfer",
    monitor_metrics=["eval_f1", "eval_precision", "eval_recall", "eval_topk_mean_acc"]
)

retrieval_callback = EmbeddingRetrievalMonitorCallback(
    base_dir=base_path / "outputs" / "goemotions_notransfer",
    dataset=val_dataset,
    ks=[1, 3, 6]
)

epoch_checkpoint_callback = EpochCheckpointSaverCallback(
    save_dir=base_path / "outputs" / "goemotions_notransfer",
    save_epochs=[3]  # Save at epoch 3
)

# ===============================
# Trainer setup
# ===============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[
        RealTimeLoggerCallback(base_path / "outputs" / "goemotions_notransfer" / "loss_log.jsonl"),
        multi_metric_callback,
        retrieval_callback,
        epoch_checkpoint_callback  # <-- Added here
    ]
)

# Inject the trainer into callbacks
multi_metric_callback.trainer = trainer
retrieval_callback.trainer = trainer

# ===============================
# Start training and final evaluation
# ===============================
trainer.train()

# Final test set evaluation
test_results = trainer.evaluate(test_dataset)
with open(base_path / "outputs" / "goemotions_notransfer" / "test_results.json", "w") as f:
    json.dump(test_results, f, indent=2)
