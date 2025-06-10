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
import os

# Create logs and outputs directory if needed
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Create a timestamped log file for JSONL
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_log_path = output_dir / f"phase_one{timestamp}.jsonl"

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

def load_va_matrix(nrc_vad_path, emotion_class_words):
    """
    Build a 28x2 VA matrix and then compute cosine similarity between emotion classes.
    
    Args:
        nrc_vad_path (str): Path to NRC-VAD-Lexicon.txt
        emotion_class_words (List[str]): 28 class labels mapped to representative words
    
    Returns:
        torch.Tensor: (28, 28) cosine similarity matrix between emotion classes
    """
    # Load the VAD lexicon into a dictionary: word -> (valence, arousal)
    vad_dict = {}
    with open(nrc_vad_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            word, valence, arousal, _ = parts
            vad_dict[word.lower()] = (float(valence), float(arousal))

    # Extract VA vectors for your 28 emotion classes
    va_vectors = []
    for word in emotion_class_words:
        if word.lower() not in vad_dict:
            raise ValueError(f"Word '{word}' not found in NRC-VAD lexicon.")
        va_vectors.append(vad_dict[word.lower()])

    # Convert to tensor: (28, 2)
    va_tensor = torch.tensor(va_vectors, dtype=torch.float32)

    # Normalize and compute cosine similarity: (28, 28)
    va_tensor = F.normalize(va_tensor, p=2, dim=1)
    similarity_matrix = torch.matmul(va_tensor, va_tensor.T)

    return similarity_matrix

# Modify the loss function for phase 1
class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        return self.bce_loss(logits, labels)
    
 
class MultiLabelContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, va_matrix: torch.Tensor = None):
        super().__init__()
        self.temperature = temperature
        self.va_matrix = va_matrix  # shape (C, C), cosine similarity between classes

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (N, D) L2-normalized
        labels: (N, C) binary multi-label (0/1)
        """
        device = embeddings.device
        N, C = labels.shape

        # 1. Contrastive Setup
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        logits_mask = ~torch.eye(N, dtype=torch.bool, device=device)  # Mask out the diagonal (self-pair)
        label_sets = [set(torch.nonzero(lbl, as_tuple=True)[0].tolist()) for lbl in labels]

        losses = []
        for i in range(N):
            # Replace sets with boolean values indicating whether there's an intersection
            pos_mask = torch.tensor([i != j and len(label_sets[i].intersection(label_sets[j])) > 0 for j in range(N)], device=device)

            if pos_mask.sum() == 0:
                continue

            positives = sim_matrix[i][pos_mask]
            negatives = sim_matrix[i][logits_mask[i]]

            # Optional: VA-based soft weights
            if self.va_matrix is not None:
                va_weights = []
                for j in range(N):
                    if i == j:
                        continue
                    shared = label_sets[i].intersection(label_sets[j])
                    if shared:
                        va_weights.append(1.0)
                    else:
                        sim = max(self.va_matrix[a][b] for a in label_sets[i] for b in label_sets[j])
                        va_weights.append(sim)
                va_weights = torch.tensor(va_weights, dtype=torch.float, device=device)
            else:
                raise ValueError("VA matrix is not provided for soft weighting.")

            # Stable log-sum-exp trick
            c = torch.max(negatives).detach()
            pos_exp = torch.exp(positives - c).sum()
            neg_exp = (torch.exp(negatives - c) * va_weights).sum()
            loss_i = -torch.log(pos_exp / (neg_exp + 1e-8))  # Only contrastive loss (no BCE)
            losses.append(loss_i)

        contrastive_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
        return contrastive_loss
    
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

@torch.no_grad()
def evaluate_embeddings_macro_precision(model, dataset, device, batch_size=128, top_k=5, num_classes=28):
    model.eval()

    all_embeddings = []
    all_labels = []

    # DataLoader to fetch batches from the dataset
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Get the embeddings from the model
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2-normalize the embeddings

        all_embeddings.append(embeddings)
        all_labels.append(labels)

    # Concatenate all embeddings and labels
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute similarity matrix (cosine similarity)
    sim_matrix = torch.matmul(all_embeddings, all_embeddings.T)
    sim_matrix.fill_diagonal_(-float('inf'))  # Exclude self-similarity

    # Get the top-k indices
    topk_indices = torch.topk(sim_matrix, k=top_k, dim=1).indices  # (N, top_k)

    # Initialize per-class precision dictionary
    per_class_precisions = {c: [] for c in range(num_classes)}

    total_samples = all_labels.size(0)
    total_hits = 0  # To count the total number of successful hits across all samples

    # Loop through all samples to calculate precision
    for i in range(total_samples):
        query_label = all_labels[i]  # shape: (num_classes,)
        retrieved_labels = all_labels[topk_indices[i]]  # shape: (top_k, num_classes)

        # Check for a match: a match is considered if at least one retrieved label matches the true label
        correct_count = (retrieved_labels * query_label).sum(dim=1) > 0  # Any overlap with true labels
        precision_i = correct_count.sum().item() / top_k  # Precision for this sample

        total_hits += correct_count.sum().item()

        # Distribute the precision across all relevant classes
        for c in torch.nonzero(query_label, as_tuple=False).squeeze(1).tolist():
            per_class_precisions[c].append(precision_i)

    # Average precision per class
    class_avg_precisions = []
    for c in range(num_classes):
        class_precisions = per_class_precisions[c]
        if len(class_precisions) > 0:
            avg_precision_c = sum(class_precisions) / len(class_precisions)
        else:
            avg_precision_c = 0.0
        class_avg_precisions.append(avg_precision_c)

    # Calculate macro average precision
    macro_avg_precision = sum(class_avg_precisions) / num_classes

    # Log the results
    logging.info(f"Per-Class Average Precisions: {class_avg_precisions}")
    logging.info(f"Macro-Averaged Top-{top_k} Precision: {macro_avg_precision:.4f}")

    # Log total hits across all samples for debugging purposes
    logging.info(f"Total successful hits: {total_hits}/{total_samples * top_k}")

    return macro_avg_precision, class_avg_precisions  # Return per-class precision for further analysis

@torch.no_grad()
def evaluate_classification_accuracy(model, dataset, device, batch_size=128, num_classes=28):
    model.eval()

    per_class_correct = {i: 0 for i in range(num_classes)}
    per_class_total = {i: 0 for i in range(num_classes)}

    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits)
        predicted_labels = (probs > 0.5).float()

        for i in range(num_classes):
            per_class_total[i] += labels[:, i].sum().item()
            per_class_correct[i] += (predicted_labels[:, i] * labels[:, i]).sum().item()

    per_class_accuracy = {
        i: per_class_correct[i] / (per_class_total[i] + 1e-8)
        for i in range(num_classes)
    }

    # NEW: macro average accuracy
    macro_accuracy = sum(per_class_accuracy.values()) / num_classes

    logging.info(f"Classification Accuracy (macro): {macro_accuracy:.4f}")
    logging.info(f"Per-Class Classification Accuracy: {per_class_accuracy}")

    return macro_accuracy, per_class_accuracy

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
        return F.normalize(x, p=2, dim=1)  # L2-normalization for contrastive learning

class EmotionEmbeddingModel(nn.Module):
    def __init__(self, dropout_rate: float = 0.3, projection_dim: int = 128, num_classes: int = 28, model_path=None, freeze_encoder: bool = True):
        super().__init__()
        
        # Load model and tokenizer from local path
        if model_path is None:
            model_path = Path(__file__).resolve().parent.parent / "outputs" / "goemotions_transfer2" / "checkpoint-12950"
        
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

        # Dual heads
        self.projection_head = ProjectionHead(
            input_dim=hidden_size,
            hidden_dim=256,
            output_dim=projection_dim,
            dropout=dropout_rate
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = self.dropout(outputs.last_hidden_state[:, 0])  # [CLS] token

        # Two outputs
        projected = self.projection_head(cls_embedding)
        return projected

# Function to log to JSONL
def log_to_jsonl(log_entry, log_file="outputs/nt_xent_va_sampler2/nt_xent_va_sampler2.jsonl"):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def train(model, train_loader, val_loader, contrastive_loss_fn, device, total_steps=10000, log_interval=10):
    step = 0
    best_macro_precision = 0.0

    # Initialize GradScaler for mixed precision training
    scaler = torch.amp.GradScaler()

    # Separate parameter groups for classification and projection heads
    projection_params = list(model.projection_head.parameters())

    # Create optimizers for the two parts of the model
    optimizer_projection = torch.optim.Adam(projection_params, lr=5e-5)

    model.train()
    logging.info(f"Training started. Total steps: {total_steps}")

    while step < total_steps:
        for batch in tqdm(train_loader, desc=f"Training Step {step + 1}", ncols=100):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer_projection.zero_grad()

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):  # Use mixed precision for GPU
                embeddings = model(input_ids, attention_mask)
                contrastive_loss = contrastive_loss_fn(embeddings, labels)  # Loss for projection head

                # Update projection head only
                scaler.scale(contrastive_loss).backward()
                scaler.step(optimizer_projection)
                scaler.update()

            # Log the loss every log_interval steps (only log train loss)
            if (step + 1) % log_interval == 0:
                logging.info(
                    f"Step {step + 1} - Contrastive Loss: {contrastive_loss.item():.4f}"
                )

            # Validate every 10 minibatches
            if (step + 1) % 10 == 0:
                # Evaluate macro precision (for contrastive learning)
                val_macro_precision, val_class_avg_precisions = evaluate_embeddings_macro_precision(model, val_loader.dataset, device, batch_size=64, top_k=5, num_classes=28)

                # Save the best model based on validation macro precision or classification accuracy
                if val_macro_precision > best_macro_precision:
                    best_macro_precision = val_macro_precision
                    torch.save(model.projection_head.state_dict(), f"outputs/nt_xent_va_sampler2/best_projection_head.pt")  # Save projection head based on macro precision
                    logging.info(f"Best projection head saved with macro precision: {val_macro_precision:.4f}")

                # Prepare log entry for validation
                log_entry = {
                    "step": step + 1,
                    "log_contrastive_loss": contrastive_loss.item(),
                    "val_macro_precision": val_macro_precision,
                    "val_class_avg_precisions": val_class_avg_precisions,
                    "timestamp": datetime.datetime.now().isoformat()
                }

                # Log to the JSONL file
                log_to_jsonl(log_entry)

            step += 1
            if step >= total_steps:
                break

    # Final log entry after training completes
    logging.info(f"Training complete. Best macro precision achieved: {best_macro_precision:.4f}")
    logging.info(f"Best projection and classification heads saved.")

# Define paths and configurations
train_loader = create_balanced_loader(train_dataset, batch_size=32, num_classes=28)
val_loader = create_balanced_loader(val_dataset, batch_size=32, num_classes=28)

# Initialize the model
model = EmotionEmbeddingModel(dropout_rate=0.3, projection_dim=128, num_classes=28, model_path=model_path, freeze_encoder=True).to(device)

# This must be your exact class-to-word mapping
goemotions_class_words = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

va_matrix = load_va_matrix("scripts/NRC-VAD-Lexicon.txt", goemotions_class_words)

# Initialize loss functions
contrastive_loss_fn = MultiLabelContrastiveLoss(temperature=0.07, va_matrix= va_matrix)
# Set up the optimizer for the model
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Start training
train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    contrastive_loss_fn=contrastive_loss_fn,
    device=device,
    total_steps=10000,  # Define the number of steps you want to train
    log_interval=10   # Interval for logging
)
