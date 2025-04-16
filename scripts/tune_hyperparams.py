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


# Create logs directory if needed
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create a timestamped log file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = log_dir / f"optuna_tuning_{timestamp}.log"

# Redirect all stdout to file (plus console if needed)
sys.stdout = open(log_path, "w", encoding="utf-8")


print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Determine the device
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

class EmotionDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=128):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item["split"] == "train":  # use only training data
                    self.samples.append(item)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = [s["label"] for s in self.samples]

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoded = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": item["label"]
        }

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return self.labels

@torch.no_grad()
def compute_validation_loss(model, dataset, criterion, num_classes=28, batch_size=32, max_steps=30):
    model.eval()

    # Build balanced sampler
    val_labels = dataset.get_labels()
    val_sampler = UniformBalancedBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        num_classes=num_classes,
        labels=val_labels
    )

    total_loss = 0.0
    steps = 0
    sampler_iter = iter(val_sampler)

    for _ in range(max_steps):
        batch_indices = next(sampler_iter)
        batch = [dataset[i] for i in batch_indices]

        input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
        attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(device)
        labels = torch.tensor([item["label"] for item in batch]).to(device)

        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(embeddings, labels)

        total_loss += loss.item()
        steps += 1

    return total_loss / steps


from transformers import AutoTokenizer

# Load tokenizer and dataset
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

base_dir = Path(__file__).resolve().parent.parent
dataset_path = base_dir / "data" / "augmented_single_label.jsonl"

train_dataset = EmotionDataset(dataset_path, tokenizer)
train_labels = train_dataset.get_labels()

# Load validation data (same tokenizer)
val_dataset = EmotionDataset(dataset_path, tokenizer)
val_samples = [s for s in val_dataset.samples if s["split"] == "validation"]
val_dataset.samples = val_samples
val_dataset.labels = [s["label"] for s in val_samples]

class EmotionEmbeddingModel(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        # Forward through encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding as sentence representation
        cls_embedding = outputs.last_hidden_state[:, 0]  # shape: (batch_size, hidden_dim)
        return cls_embedding

model = EmotionEmbeddingModel().to(device)

def train_and_evaluate(hparams, train_dataset, val_dataset, device, num_classes=28, steps_per_epoch=100, val_steps=30):
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch

    # 1. Setup
    batch_size = hparams["batch_size"]
    learning_rate = hparams["learning_rate"]
    temperature = hparams["temperature"]

    model = EmotionEmbeddingModel().to(device)
    criterion = BSCLossSingleLabel(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 2. Sampler & Iterators
    train_labels = train_dataset.get_labels()
    train_sampler = UniformBalancedBatchSampler(
        dataset=train_dataset,
        batch_size=batch_size,
        num_classes=num_classes,
        labels=train_labels
    )
    train_sampler_iter = iter(train_sampler)

    val_labels = val_dataset.get_labels()
    val_sampler = UniformBalancedBatchSampler(
        dataset=val_dataset,
        batch_size=batch_size,
        num_classes=num_classes,
        labels=val_labels
    )
    val_sampler_iter = iter(val_sampler)


    # 3. Training Loop
    num_epochs = hparams.get("num_epochs", 3)
    print_every = hparams.get("print_every", 10)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        print(f"\n==== Epoch {epoch + 1} ====")
        for step in range(steps_per_epoch):
            batch_indices = next(train_sampler_iter)
            batch = [train_dataset[i] for i in batch_indices]

            input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
            attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(device)
            labels = torch.tensor([item["label"] for item in batch]).to(device)

            optimizer.zero_grad()
            embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % print_every == 0:
                print(f"Step {step:03d} | Train Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / steps_per_epoch
        avg_val_loss = compute_validation_loss(model, val_sampler_iter, val_dataset, criterion, steps=val_steps)
        print(f"Epoch {epoch + 1} — Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return avg_val_loss  # Use for Optuna or score-based model selection

import optuna

def objective(trial):
    hparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        "temperature": trial.suggest_float("temperature", 0.03, 0.2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "num_epochs": 3,
        "print_every": 10
    }

    val_loss = train_and_evaluate(
        hparams=hparams,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device
    )

    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Final results
print("Best trial:")
print(f"Validation Loss: {study.best_value:.4f}")
print("Hyperparameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

Path("outputs").mkdir(exist_ok=True)
with open(f"outputs/best_params_{timestamp}.json", "w", encoding="utf-8") as f:
    json.dump({
        "best_value": study.best_value,
        "best_params": study.best_params
    }, f, indent=2)

print("Done and saved best parameters.")