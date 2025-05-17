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
    

class JointContrastiveClassificationLoss(nn.Module):
    def __init__(self, temperature=0.07, alpha=0.1, va_matrix=None, use_entropy_weight=True, va_smooth_power=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.va_matrix = va_matrix  # shape (C, C)
        self.use_entropy_weight = use_entropy_weight
        self.va_smooth_power = va_smooth_power  # Apply sim^β to reduce aggressive negatives

    def forward(self, embeddings, logits, labels):
        device = embeddings.device
        N, C = labels.shape

        # Binary cross entropy classification loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        # Entropy-based instance weights
        if self.use_entropy_weight:
            probs = torch.sigmoid(logits)
            entropy = - (probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
            label_sums = labels.sum(dim=1)
            safe_entropy = (entropy * labels).sum(dim=1) / torch.clamp(label_sums, min=1.0)
            entropy_weight = 1.0 - (safe_entropy / torch.log(torch.tensor(2.0)).to(device))  # Normalize to [0,1]
            entropy_weight = torch.clamp(entropy_weight, min=0.0, max=1.0)
        else:
            entropy_weight = torch.ones(N, device=device)

        # Contrastive setup
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        logits_mask = ~torch.eye(N, dtype=torch.bool, device=device)
        label_sets = [set(torch.nonzero(lbl, as_tuple=True)[0].tolist()) for lbl in labels]

        contrastive_losses = []
        for i in range(N):
            pos_mask = torch.tensor(
                [i != j and bool(label_sets[i] & label_sets[j]) for j in range(N)],
                dtype=torch.bool, device=device
            )

            if pos_mask.sum() == 0:
                continue

            positives = sim_matrix[i][pos_mask]
            neg_indices = torch.where(logits_mask[i])[0]
            negatives = sim_matrix[i][neg_indices]

            # Build VA weights only for negatives
            if self.va_matrix is not None:
                va_weights = []
                for j in neg_indices.tolist():
                    if label_sets[i] & label_sets[j]:
                        va_weights.append(1.0)
                    else:
                        sim = max(self.va_matrix[a][b] for a in label_sets[i] for b in label_sets[j])
                        sim = sim ** self.va_smooth_power  # Optional smoothing
                        va_weights.append(sim)
                va_weights = torch.tensor(va_weights, device=device, dtype=torch.float)
            else:
                raise ValueError("VA matrix is not provided.")

            # Stable log-sum-exp
            c = torch.max(negatives).detach()
            pos_exp = torch.exp(positives - c).sum()
            neg_exp = (torch.exp(negatives - c) * va_weights).sum()

            loss_i = -torch.log(pos_exp / (neg_exp + 1e-8)) * entropy_weight[i]
            contrastive_losses.append(loss_i)

        contrastive_loss = torch.stack(contrastive_losses).mean() if contrastive_losses else torch.tensor(0.0, device=device)
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
        embeddings, logits = model(input_ids=input_ids, attention_mask=attention_mask)
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
    def __init__(self,
                 dropout_rate: float = 0.3,
                 projection_dim: int = 128,
                 num_classes: int = 28,
                 model_path=None,
                 freeze_encoder: bool = True,
                 pretrained_projection_path: Optional[Union[str, Path]] = None,
                 pretrained_classification_path: Optional[Union[str, Path]] = None):
        super().__init__()

        # Default encoder path if none provided
        if model_path is None:
            model_path = Path(__file__).resolve().parent.parent / "models" / "roberta-base-go_emotions"
        
        # Load encoder
        self.encoder = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout_rate)

        hidden_size = self.encoder.config.hidden_size

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Projection Head
        self.projection_head = ProjectionHead(
            input_dim=hidden_size,
            hidden_dim=256,
            output_dim=projection_dim,
            dropout=dropout_rate
        )

        # Classification Head
        self.classification_head = ClassificationHead(
            input_dim=hidden_size,
            hidden_dim=256,
            num_classes=num_classes,
            dropout=dropout_rate
        )

        # Load pretrained heads if provided
        if pretrained_projection_path and os.path.exists(pretrained_projection_path):
            self.projection_head.load_state_dict(torch.load(pretrained_projection_path, map_location=device))
            logging.info(f"Loaded projection head from {pretrained_projection_path}")
        else:
            raise ValueError(f"Pretrained projection head not found at {pretrained_projection_path}")

        if pretrained_classification_path and os.path.exists(pretrained_classification_path):
            self.classification_head.load_state_dict(torch.load(pretrained_classification_path, map_location=device))
            logging.info(f"Loaded classification head from {pretrained_classification_path}")
        else:
            raise ValueError(f"Pretrained classification head not found at {pretrained_classification_path}")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = self.dropout(outputs.last_hidden_state[:, 0])
        projected = self.projection_head(cls_embedding)
        logits = self.classification_head(cls_embedding)
        return projected, logits

# Function to log to JSONL
def log_to_jsonl(log_entry, log_file="outputs/stage_two_trainning/stage_two_trainning.jsonl"):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def train_stage2_joint_loss(model, train_loader, val_loader, loss_fn, device, total_steps=10000, log_interval=10):
    step = 0
    best_macro_precision = 0.0
    best_classification_accuracy = 0.0

    scaler = torch.amp.GradScaler()

    # Optimizer now updates both heads jointly
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    logging.info(f"[Stage 2] Joint training started for {total_steps} steps.")

    while step < total_steps:
        for batch in tqdm(train_loader, desc=f"[Step {step+1}]", ncols=100):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                embeddings, logits = model(input_ids, attention_mask)
                loss = loss_fn(embeddings, logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (step + 1) % log_interval == 0:
                logging.info(f"[Step {step+1}] Joint Loss: {loss.item():.4f}")

            # Validation & checkpointing
            if (step + 1) % 10 == 0:
                val_macro_precision, val_class_avg_precisions = evaluate_embeddings_macro_precision(
                    model, val_loader.dataset, device, batch_size=64, top_k=5, num_classes=28
                )
                val_classification_accuracy, val_class_accuracy = evaluate_classification_accuracy(
                    model, val_loader.dataset, device, batch_size=64, num_classes=28
                )

                # Save best checkpoint by retrieval quality
                if val_macro_precision > best_macro_precision:
                    best_macro_precision = val_macro_precision
                    torch.save(model.state_dict(), f"outputs/stage_two_trainning/best_model_retrieval.pt")
                    logging.info(f"[Step {step+1}] Best retrieval model saved: {val_macro_precision:.4f}")

                # Save best checkpoint by classification
                if val_classification_accuracy > best_classification_accuracy:
                    best_classification_accuracy = val_classification_accuracy
                    torch.save(model.state_dict(), f"outputs/stage_two_trainning/best_model_classification.pt")
                    logging.info(f"[Step {step+1}] Best classification model saved: {val_classification_accuracy:.4f}")

                # JSONL logging
                log_to_jsonl({
                    "step": step + 1,
                    "joint_loss": loss.item(),
                    "val_macro_precision": val_macro_precision,
                    "val_classification_accuracy": val_classification_accuracy,
                    "val_class_avg_precisions": val_class_avg_precisions,
                    "val_class_accuracy": val_class_accuracy,
                    "timestamp": datetime.datetime.now().isoformat()
                })

            step += 1
            if step >= total_steps:
                break

    logging.info(f"[Stage 2] Training complete. Best macro precision: {best_macro_precision:.4f}")
    logging.info(f"[Stage 2] Best classification accuracy: {best_classification_accuracy:.4f}")


# Define paths and configurations
train_loader = create_balanced_loader(train_dataset, batch_size=32, num_classes=28)
val_loader = create_balanced_loader(val_dataset, batch_size=32, num_classes=28)


# Resolve the absolute path to the script's directory
script_dir = Path(__file__).resolve().parent

# Use `.resolve()` again to normalize the path after ".."
pretrained_projection_path = (script_dir / ".." / "outputs" / "stage_one_trainning_va" / "best_projection_head.pt").resolve()
pretrained_classification_path = (script_dir / ".." / "outputs" / "stage_one_trainning_va" / "best_classification_head.pt").resolve()


model = EmotionEmbeddingModel(
    dropout_rate=0.3, 
    projection_dim=128, 
    num_classes=28, 
    pretrained_projection_path = pretrained_projection_path,
    pretrained_classification_path = pretrained_classification_path,
    freeze_encoder = False
).to(device)

# This must be your exact class-to-word mapping
goemotions_class_words = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

va_matrix = load_va_matrix("scripts/NRC-VAD-Lexicon.txt", goemotions_class_words)

loss_fn = JointContrastiveClassificationLoss(
    temperature=0.07,
    alpha=1.0,
    va_matrix=va_matrix,                # or pass the VA matrix if loaded
    use_entropy_weight=True
)

train_stage2_joint_loss(  
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,  # pass the full loss that returns bce + alpha * contrastive
    device=device,
    total_steps=10000,
    log_interval=10
)
