import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import random
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


@torch.no_grad()
def compute_validation_loss(model, dataset, criterion, batch_size=32):
    model.eval()
    val_loader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0.0
    steps = 0

    for batch in val_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
        loss       = criterion(embeddings, labels)

        total_loss += loss.item()
        steps += 1

    return total_loss / steps

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


class EmotionEmbeddingModel(nn.Module):
    def __init__(self, model_dir: Path = None):
        super().__init__()
        # if no path passed, default to your local “model” directory one level
        if model_dir is None:
            model_dir = Path(__file__).resolve().parent.parent / "model"
        # force local‑only load
        self.encoder = AutoModel.from_pretrained(
            str(model_dir),
            local_files_only=True
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # take [CLS] embedding
        return outputs.last_hidden_state[:, 0]

model = EmotionEmbeddingModel().to(device)

def train_and_evaluate(hparams, train_dataset, val_dataset, device,
                       num_classes=28, steps_per_epoch=400):
    model = EmotionEmbeddingModel().to(device)
    criterion = BSCLossSingleLabel(temperature=hparams["temperature"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["learning_rate"])

    # only need a sampler for training
    train_sampler = UniformBalancedBatchSampler(
        dataset    = train_dataset,
        batch_size = hparams["batch_size"],
        num_classes= num_classes,
        labels     = train_dataset.get_labels()
    )
    train_it = iter(train_sampler)

    for epoch in range(hparams["num_epochs"]):
        model.train()
        total_loss = 0.0

        print(f"\n==== Epoch {epoch + 1} ====")
        for step in range(steps_per_epoch):
            batch_indices = next(train_it)
            batch = [train_dataset[i] for i in batch_indices]

            input_ids      = torch.stack([b["input_ids"]      for b in batch]).to(device)
            attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(device)
            labels         = torch.tensor([b["label"] for b in batch]).to(device)

            optimizer.zero_grad()
            embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
            loss       = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % hparams["print_every"] == 0:
                print(f"Step {step:03d} | Train Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / steps_per_epoch

        # VALIDATE over the *entire* val set
        avg_val_loss = compute_validation_loss(
            model, val_dataset, criterion,
            batch_size=hparams["batch_size"]
        )

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", flush=True)

    return avg_val_loss

def objective(trial):
    hparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        "temperature": trial.suggest_float("temperature", 0.03, 0.2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "num_epochs": 4,
        "print_every": 100
    }

    val_loss = train_and_evaluate(
        hparams=hparams,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device
    )

    # Save trial summary *before* returning
    with open(output_log_path, "a", encoding="utf-8") as f:
        json.dump({
            "trial_number": trial.number,
            "params": trial.params,
            "val_loss": val_loss
        }, f)
        f.write("\n")


    return val_loss

study = optuna.create_study(direction="minimize")
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