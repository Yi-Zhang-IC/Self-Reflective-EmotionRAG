import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import random
import time
from torch.utils.data import Sampler
from collections import defaultdict
from pathlib import Path
import sys
import datetime
from pathlib import Path
import json
from torch.utils.data import Dataset
import optuna
from typing import Union, Optional
import torch.nn.init as init
from torch.cuda.amp import autocast, GradScaler
from torch import amp
from torch.cuda.amp import autocast, GradScaler


# Create logs and outputs directory if needed
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Create a timestamped log file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = log_dir / f"optuna_tuning_{timestamp}.log"
output_log_path = output_dir / f"trial_log_{timestamp}.jsonl"

# Redirect stdout to the log file BEFORE any prints
sys.stdout = open(log_path, "w", encoding="utf-8", buffering=1)
sys.stderr = sys.stdout

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class UniformBalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_classes, labels):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.labels = labels

        # Build mapping: class -> list of indices
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        self.available_classes = [c for c in range(self.num_classes) if len(self.class_to_indices[c]) > 0]

        if self.batch_size < len(self.available_classes):
            raise ValueError(f"Batch size must be >= number of available classes ({len(self.available_classes)}). Got {self.batch_size}.")

    def __iter__(self):
        while True:
            batch_indices = []

            # Sample one from each available class
            for class_id in self.available_classes:
                candidates = self.class_to_indices[class_id]
                if candidates:
                    selected = random.choice(candidates)
                    batch_indices.append(selected)

            # Fill remaining slots randomly
            while len(batch_indices) < self.batch_size:
                class_id = random.choice(self.available_classes)
                candidates = self.class_to_indices[class_id]
                selected = random.choice(candidates)
                batch_indices.append(selected)

            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return 1000000  # or some large number

class BSCLossSingleLabel(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super(BSCLossSingleLabel, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: Tensor of shape (batch_size, hidden_dim) - embeddings
            labels: Tensor of shape (batch_size,) - single integer label per sample
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features to unit hypersphere
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix (batch_size x batch_size)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Positive mask: label[i] == label[j], and i != j
        labels = labels.view(-1, 1)  # shape (batch_size, 1)
        positive_mask = (labels == labels.T).float()  # (batch_size, batch_size)
        diag_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask * (1 - diag_mask)  # remove diagonal

        losses = []

        for i in range(batch_size):
            pos_indices = positive_mask[i].nonzero(as_tuple=False).squeeze(1)

            if len(pos_indices) == 0:
                continue

            numerator = torch.exp(sim_matrix[i, pos_indices])

            denominator = 0.0
            for c in torch.unique(labels):
                class_indices = (labels.squeeze() == c).nonzero(as_tuple=False).squeeze(1)
                class_indices = class_indices[class_indices != i]

                if len(class_indices) == 0:
                    continue

                class_sims = torch.exp(sim_matrix[i, class_indices])
                class_sum = class_sims.sum()
                class_sum = class_sum / len(class_indices)

                denominator += class_sum

            loss_i = - torch.mean(torch.log(numerator / denominator))
            losses.append(loss_i)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)

        return torch.mean(torch.stack(losses))

def log_trial_result(trial_id, epoch, step, train_loss=None, val_loss=None):
    log_file = Path("outputs/training_log.jsonl")
    log_entry = {
        "trial": trial_id,
        "epoch": epoch,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def create_balanced_loader(dataset, batch_size, num_classes):
    sampler = UniformBalancedBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        num_classes=num_classes,
        labels=dataset.get_labels()
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=8,                  # Use 8 CPU workers (adjust based on your machine)
        pin_memory=True,               # Speeds up CPU→GPU transfer
        persistent_workers=True,       # Keep workers alive between epochs
        prefetch_factor=4              # Preload 4 batches ahead
    )


class EmotionDataset(Dataset):
    def __init__(self,
                 path: Union[Path,str],
                 tokenizer,
                 split: Optional[str] = None,
                 max_length: int = 128):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # if split is None, take all; otherwise only items matching that split
                if split is None or item["split"] == split:
                    self.samples.append(item)

        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.labels     = [s["label"] for s in self.samples]

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc  = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          item["label"]
        }

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return self.labels

# 1) point at your local “model” folder
model_path = Path(__file__).resolve().parent.parent / "model"
tokenizer  = AutoTokenizer.from_pretrained(model_path)

# 2) point at the single JSONL that already has 'split' fields
base_dir     = Path(__file__).resolve().parent.parent
dataset_path = base_dir / "data" / "augmented_single_label.jsonl"

# 3) load train/val by passing split="train" or split="validation"
train_dataset = EmotionDataset(dataset_path, tokenizer, split="train")
val_dataset   = EmotionDataset(dataset_path, tokenizer, split="validation")

train_labels = train_dataset.get_labels()
val_labels   = val_dataset.get_labels()

@torch.no_grad()
def evaluate_embeddings_macro_precision(model, dataset, device, batch_size=64, top_k=5, num_classes=28):
    model.eval()

    all_embeddings = []
    all_labels = []

    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings)
        all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    sim_matrix = torch.matmul(all_embeddings, all_embeddings.T)
    sim_matrix.fill_diagonal_(-float('inf'))  # Exclude self-similarity

    topk_indices = torch.topk(sim_matrix, k=top_k, dim=1).indices  # (N, top_k)

    per_class_precisions = {c: [] for c in range(num_classes)}

    total_samples = all_labels.size(0)

    for i in range(total_samples):
        query_label = all_labels[i].item()
        retrieved_labels = all_labels[topk_indices[i]]

        correct_count = (retrieved_labels == query_label).sum().item()
        precision_i = correct_count / top_k

        per_class_precisions[query_label].append(precision_i)

    class_avg_precisions = []
    for c in range(num_classes):
        class_precisions = per_class_precisions[c]
        if len(class_precisions) > 0:
            avg_precision_c = sum(class_precisions) / len(class_precisions)
        else:
            avg_precision_c = 0.0
        class_avg_precisions.append(avg_precision_c)

    macro_avg_precision = sum(class_avg_precisions) / num_classes

    print(f"Per-Class Average Precisions: {class_avg_precisions}")
    print(f"Macro-Averaged Top-{top_k} Precision: {macro_avg_precision:.4f}")

    return macro_avg_precision

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # You could also use kaiming_uniform_ or normal_
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=1)  # Normalize to unit sphere (important for cosine-based contrastive loss)

class EmotionEmbeddingModel(nn.Module):
    def __init__(self, model_dir: Path = None, dropout_rate: float = 0.3, projection_dim: int = 128):
        super().__init__()
        if model_dir is None:
            model_dir = Path(__file__).resolve().parent.parent / "model"
        
        # Load the pre-trained encoder
        self.encoder = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device)  

        self.dropout = nn.Dropout(dropout_rate)
        
        # Add projection head (input_dim depends on encoder hidden size, usually 768)
        hidden_size = self.encoder.config.hidden_size  # Automatically pick correct size
        self.projection_head = ProjectionHead(
            input_dim=hidden_size,
            hidden_dim=256,
            output_dim=projection_dim,
            dropout=dropout_rate
        )
        self.projection_head.apply(init_weights)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = self.dropout(outputs.last_hidden_state[:, 0])
        projected = self.projection_head(cls_embeddings)
        return projected

def warmup_train(hparams, train_dataset, val_dataset, device, num_classes=28, steps_per_epoch=600, num_epochs=10, save_path="warmup_checkpoint.pt"):
    model = EmotionEmbeddingModel(
        dropout_rate=hparams["dropout_rate"],
        projection_dim=hparams.get("projection_dim", 128)
    ).to(device)

    model = torch.compile(model, backend="eager")
    criterion = BSCLossSingleLabel(temperature=hparams["temperature"])
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": hparams["encoder_lr"]},
        {"params": model.projection_head.parameters(), "lr": hparams["head_lr"]}
    ], weight_decay=1e-5)
    scaler = GradScaler()

    train_loader = create_balanced_loader(
        dataset=train_dataset,
        batch_size=hparams["batch_size"],
        num_classes=num_classes
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        print(f"\n==== Warmup Epoch {epoch + 1} ====")
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(embeddings, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if step >= steps_per_epoch:
                break

        avg_loss = total_loss / steps_per_epoch
        print(f"[Warmup Epoch {epoch+1}] Average Train Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"Saved warmup checkpoint to {save_path}")

def train_and_evaluate(hparams, train_dataset, val_dataset, device, trial,
                       num_classes=28, steps_per_epoch=600):

    model = EmotionEmbeddingModel(
        dropout_rate=hparams["dropout_rate"],
        projection_dim=hparams.get("projection_dim", 128)
    ).to(device)

    # Apply torch.compile for speedup
    model = torch.compile(model, backend="eager")

    criterion = BSCLossSingleLabel(temperature=hparams["temperature"])
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": hparams["encoder_lr"]},
        {"params": model.projection_head.parameters(), "lr": hparams["head_lr"]}
    ], weight_decay=1e-5)

    scaler = amp.GradScaler()  # Correct initialization (no device_type argument)

    # Use your balanced sampler / loader
    train_loader = create_balanced_loader(
        dataset=train_dataset,
        batch_size=hparams["batch_size"],
        num_classes=num_classes
    )

    val_counter = 0  # Optuna pruning counter

    for epoch in range(hparams["num_epochs"]):
        model.train()
        total_loss = 0.0
        print(f"\n==== Epoch {epoch + 1} ====")

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with amp.autocast(device_type='cuda', dtype=torch.float16):
                embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(embeddings, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"Step {step:03d} | Train Loss: {loss.item():.4f}", flush=True)

            if step > 0 and step % 200 == 0:
                model.eval()
                start_time = time.time()
                val_metric = evaluate_embeddings_macro_precision(
                    model=model,
                    dataset=val_dataset,
                    device=device,
                    batch_size=hparams["batch_size"],
                    top_k=5,
                    num_classes=num_classes
                )

                val_counter += 1
                trial.report(1.0 - val_metric, val_counter)

                print(f"Step {step:03d} | 1 - Macro Precision@5: {1.0 - val_metric:.6f} | Val Time: {time.time() - start_time:.2f}s")

                if trial.should_prune():
                    print(f"[PRUNED] Trial {trial.number} stopped early at validation {val_counter}.")
                    raise optuna.TrialPruned()

                model.train()

            if step >= steps_per_epoch:
                break  # Early exit if steps_per_epoch exceeded

        avg_train_loss = total_loss / steps_per_epoch

        model.eval()
        val_metric = evaluate_embeddings_macro_precision(
            model=model,
            dataset=val_dataset,
            device=device,
            batch_size=hparams["batch_size"],
            top_k=5,
            num_classes=num_classes
        )

        val_counter += 1
        trial.report(1.0 - val_metric, val_counter)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Final 1 - Macro Precision@5: {1.0 - val_metric:.6f}")

        if trial.should_prune():
            print(f"[PRUNED] Trial {trial.number} stopped after epoch {epoch + 1}.")
            raise optuna.TrialPruned()

    return 1.0 - val_metric  # Optuna minimizes this value

def objective(trial):
    hparams = {
        "encoder_lr": trial.suggest_float("encoder_lr", 1e-6, 1e-4, log=True),
        "head_lr": trial.suggest_float("head_lr", 1e-5, 5e-4, log=True),
        "temperature": trial.suggest_float("temperature", 0.01, 0.2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }

    # Fixed hyperparameters outside Optuna search space
    hparams["projection_dim"] = 128
    hparams["num_epochs"] = 4
    hparams["print_every"] = 100
    hparams["dropout_rate"] = 0.3

    print(f"Starting Trial {trial.number} with Hyperparameters: {hparams}")

    # Train and evaluate — using macro-averaged precision@5
    val_metric = train_and_evaluate(
        hparams=hparams,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        trial=trial
    )

    print(f"Trial {trial.number} Result: 1 - Precision@5 = {val_metric:.6f}")

    # Save trial results to log file
    trial_result = {
        "trial_number": trial.number,
        "params": trial.params,
        "val_metric": val_metric  # This is 1 - macro precision@5
    }

    with open(output_log_path, "a", encoding="utf-8") as f:
        json.dump(trial_result, f)
        f.write("\n")

    return val_metric  # Optuna minimizes this (1 - macro precision@5)

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(n_startup_trials=20),
    pruner=optuna.pruners.MedianPruner(
        n_warmup_steps=4  # This means at least 4 validation reports before pruning decisions
    )
)

study.optimize(objective, n_trials=100)

# Final results
print("Best trial:")
print(f"Validation Loss: {study.best_value:.4f}")
print("Hyperparameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# Save the best parameters
with open(output_dir / f"best_params_{timestamp}.json", "w", encoding="utf-8") as f:
    json.dump({
        "best_value": study.best_value,
        "best_params": study.best_params
    }, f, indent=2)

print("Done and saved best parameters.")