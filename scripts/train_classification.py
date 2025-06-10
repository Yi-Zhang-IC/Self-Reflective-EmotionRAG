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
from torch import amp
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# Create logs and outputs directory if needed
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Set up logging for stdout (to control output)
logging.basicConfig(level=logging.INFO)

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
    def __init__(self, dataset, batch_size, num_classes, labels, base_single=2, base_multi=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.labels = labels
        self.base_single = base_single
        self.base_multi = base_multi
        self.init_goal = base_single + base_multi + 1

        self.class_to_single = defaultdict(list)
        self.class_to_multi = defaultdict(list)

        for idx, lbls in enumerate(labels):
            if isinstance(lbls, int):
                lbls = [lbls]
            if len(lbls) == 1:
                self.class_to_single[lbls[0]].append(idx)
            else:
                for l in lbls:
                    self.class_to_multi[l].append(idx)

    def __iter__(self):
        while True:
            batch_indices = set()
            class_counts = {c: 0 for c in range(self.num_classes)}

            # Phase 1: Add base_single and base_multi samples per class if available
            for c in range(self.num_classes):
                added = 0

                # Add up to base_single single-label samples
                random.shuffle(self.class_to_single[c])
                for idx in self.class_to_single[c]:
                    if idx not in batch_indices:
                        batch_indices.add(idx)
                        for l in self.labels[idx]:
                            class_counts[l] += 1
                        added += 1
                    if added >= self.base_single:
                        break

                # Add up to base_multi multi-label samples
                random.shuffle(self.class_to_multi[c])
                for idx in self.class_to_multi[c]:
                    if idx not in batch_indices:
                        batch_indices.add(idx)
                        for l in self.labels[idx]:
                            class_counts[l] += 1
                        added += 1
                    if added >= self.base_single + self.base_multi:
                        break

            # Phase 2: Iterative balancing using dynamic goal
            goal = self.init_goal
            while len(batch_indices) < self.batch_size:
                class_candidates = [c for c in range(self.num_classes) if class_counts[c] < goal]
                if not class_candidates:
                    goal += 1
                    continue

                c = random.choice(class_candidates)
                use_multi = random.random() < 0.5
                pool = self.class_to_multi[c] if use_multi else self.class_to_single[c]

                if not pool:
                    continue

                idx = random.choice(pool)
                if idx in batch_indices:
                    continue

                batch_indices.add(idx)
                for l in self.labels[idx]:
                    class_counts[l] += 1

            yield list(batch_indices)

    def __len__(self):
        return 1000000

class EmotionDataset(Dataset):
    def __init__(self, path: Union[Path, str], tokenizer, num_classes: int = 28, max_length: int = 256):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)

        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes  # Number of classes for multi-label task
        self.labels     = [s["labels"] for s in self.samples]  # Multi-label list of labels

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",  # Ensure padding is done to a fixed max_length
            return_tensors="pt",
        )

        # Handle multi-label padding
        label = item["labels"]
        label_vector = torch.zeros(self.num_classes)  # Initialize a zero vector of size num_classes
        for l in label:
            label_vector[l] = 1  # Set the class indices to 1

        return {
            "input_ids": enc["input_ids"].squeeze(0),  # Remove the batch dimension
            "attention_mask": enc["attention_mask"].squeeze(0),  # Remove the batch dimension
            "label": label_vector  # Return the multi-label as a binary vector
        }

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return self.labels

# 1) Point to your local “model” folder
model_path = Path(__file__).resolve().parent.parent / "models" / "roberta-base-go_emotions"
pretrained_classification_head_path = Path(__file__).resolve().parent.parent / "outputs" / "stage_one_trainning_va" / "best_classification_head.pt"
tokenizer = AutoTokenizer.from_pretrained(model_path)  # Correct tokenizer path

# 2) Define paths to your train, validation, and test JSONL files
base_dir = Path(__file__).resolve().parent.parent
dataset_dir = base_dir / "data" / "augmented_go_emotion"
train_path = dataset_dir / "train.jsonl"
val_path = dataset_dir / "validation.jsonl"
test_path = dataset_dir / "test.jsonl"

# 3) Load train/val/test datasets
train_dataset = EmotionDataset(train_path, tokenizer)
val_dataset   = EmotionDataset(val_path, tokenizer)
test_dataset  = EmotionDataset(test_path, tokenizer)

# Get labels for each dataset
train_labels = train_dataset.get_labels()
val_labels   = val_dataset.get_labels()
test_labels  = test_dataset.get_labels()

# Log the dataset sizes
logging.info(f"Train dataset loaded with {len(train_dataset)} samples.")
logging.info(f"Validation dataset loaded with {len(val_dataset)} samples.")
logging.info(f"Test dataset loaded with {len(test_dataset)} samples.")

class ClassificationHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=28, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)  # No sigmoid, since we use BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x)

class EmotionEmbeddingModel(nn.Module):
    def __init__(
        self,
        dropout_rate: float = 0.3,
        projection_dim: int = 128,
        num_classes: int = 28,
        model_path=None,
        freeze_encoder: bool = True,
        pretrained_classification_head_path: Union[str, Path, None] = None,
    ):
        super().__init__()

        # Load model and tokenizer from local path
        if model_path is not None:
            if not isinstance(model_path, (str, Path)):
                raise ValueError("model_path must be a string or Path object.")
            model_path = str(model_path)
        else :
            raise ValueError("model_path must be provided.")
        
        self.encoder = AutoModel.from_pretrained(model_path)  # Load from local directory
        self.dropout = nn.Dropout(dropout_rate)

        # Drop classification layer if exists in encoder
        if hasattr(self.encoder, "classifier"):
            self.encoder.classifier = nn.Identity()

        hidden_size = self.encoder.config.hidden_size

        # Freeze the encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Instantiate classification head
        self.classification_head = ClassificationHead(
            input_dim=hidden_size,
            hidden_dim=256,
            num_classes=num_classes,
            dropout=dropout_rate
        )

        # Load pretrained weights if provided
        if pretrained_classification_head_path is not None:
            logging.info(f"Loading pretrained classification head from {pretrained_classification_head_path}")
            state_dict = torch.load(pretrained_classification_head_path, map_location='cuda')
            self.classification_head.load_state_dict(state_dict)
        else:
            raise ValueError("Pretrained classification head path must be provided.")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = self.dropout(outputs.last_hidden_state[:, 0])  # [CLS] token
        logits = self.classification_head(cls_embedding)
        return logits


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

# Function to log to JSONL
def log_to_jsonl(log_entry, log_file="outputs/BCE_training/standard_bce_training.jsonl"):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def train_bce(
    model,
    train_loader,
    val_loader,
    device,
    total_steps=10000,
    log_interval=10,
    save_dir="outputs/BCE_training"
):
    os.makedirs(save_dir, exist_ok=True)
    
    step = 0
    scaler = torch.amp.GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Tracking best scores
    best_scores = {
        "accuracy": 0.0,
        "f1": 0.0,
        "recall": 0.0,
        "precision": 0.0
    }

    model.train()

    while step < total_steps:
        for batch in tqdm(train_loader, desc=f"Training Step {step + 1}", ncols=100):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (step + 1) % log_interval == 0:
                logging.info(f"Step {step + 1} - BCE Loss: {loss.item():.4f}")

            if (step + 1) % 10 == 0:
                model.eval()
                all_preds, all_targets = [], []
                for val_batch in DataLoader(val_loader.dataset, batch_size=128, shuffle=False):
                    with torch.no_grad():
                        input_ids = val_batch["input_ids"].to(device)
                        attention_mask = val_batch["attention_mask"].to(device)
                        labels = val_batch["label"].cpu().numpy()
                        logits = model(input_ids, attention_mask)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        preds = (probs > 0.5).astype(float)
                        all_preds.append(preds)
                        all_targets.append(labels)
                
                all_preds = np.concatenate(all_preds, axis=0)
                all_targets = np.concatenate(all_targets, axis=0)

                macro_precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
                macro_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
                macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

                # Macro accuracy = mean recall per class (same as earlier code)
                correct_per_class = (all_preds * all_targets).sum(axis=0)
                total_per_class = all_targets.sum(axis=0)
                macro_accuracy = (correct_per_class / (total_per_class + 1e-8)).mean()

                # Save best models by different metrics
                if macro_accuracy > best_scores["accuracy"]:
                    best_scores["accuracy"] = macro_accuracy
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_model_macro_accuracy.pt"))
                    logging.info(f"Saved best accuracy model at step {step + 1}")

                if macro_f1 > best_scores["f1"]:
                    best_scores["f1"] = macro_f1
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_model_macro_f1.pt"))
                    logging.info(f"Saved best F1 model at step {step + 1}")

                if macro_precision > best_scores["precision"]:
                    best_scores["precision"] = macro_precision
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_model_macro_precision.pt"))
                    logging.info(f"Saved best precision model at step {step + 1}")

                if macro_recall > best_scores["recall"]:
                    best_scores["recall"] = macro_recall
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_model_macro_recall.pt"))
                    logging.info(f"Saved best recall model at step {step + 1}")

                # Log to JSONL
                log_entry = {
                    "step": step + 1,
                    "bce_loss": loss.item(),
                    "val_macro_accuracy": float(macro_accuracy),
                    "val_macro_precision": float(macro_precision),
                    "val_macro_recall": float(macro_recall),
                    "val_macro_f1": float(macro_f1),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                log_to_jsonl(log_entry)
                model.train()

            step += 1
            if step >= total_steps:
                break

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    logging.info(f"Training complete. Final model saved. Best Macro Accuracy: {best_scores['accuracy']:.4f}, "
                 f"Precision: {best_scores['precision']:.4f}, Recall: {best_scores['recall']:.4f}, F1: {best_scores['f1']:.4f}")

# Define paths and configurations
train_loader = create_balanced_loader(train_dataset, batch_size=32, num_classes=28)
val_loader = create_balanced_loader(val_dataset, batch_size=32, num_classes=28)

# Initialize the model
model = EmotionEmbeddingModel(
    dropout_rate=0.3, 
    projection_dim=128, 
    num_classes=28, 
    model_path=model_path,
    pretrained_classification_head_path=pretrained_classification_head_path, 
    freeze_encoder=False
    ).to(device)

# This must be your exact class-to-word mapping
goemotions_class_words = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Set up the optimizer for the model
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Start training
train_bce(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    total_steps=10000,  # Define the number of steps you want to train
    log_interval=10   # Interval for logging
)
