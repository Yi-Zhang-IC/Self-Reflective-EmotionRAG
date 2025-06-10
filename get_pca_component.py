import torch
import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pathlib import Path
import json

# ---- User Parameters ----
encoder_path = "outputs/goemotions_notransfer/best_eval_f1_ckpt"
tokenizer_path = encoder_path
train_jsonl_path = "data/augmented_go_emotion/train.jsonl"
output_pca_path = "outputs/pca/pca_components_128.npy"
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Tokenizer & Dataset ----
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, num_classes=28, max_length=128):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }

    def __len__(self):
        return len(self.samples)

train_dataset = EmotionDataset(train_jsonl_path, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

# ---- Load Encoder ----
from transformers import AutoModel
encoder = AutoModel.from_pretrained(encoder_path).to(device)
encoder.eval()

# ---- Step 1: Extract Embeddings ----
all_embeddings = []

with torch.no_grad():
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0]  # [CLS] token
        all_embeddings.append(cls_embeddings.cpu())

X = torch.cat(all_embeddings, dim=0).numpy()  # shape: (N, 768)

# ---- Step 2: Fit PCA and Save ----
print(f"Fitting PCA on shape: {X.shape}")
pca = PCA(n_components=128)
pca.fit(X)

Path(output_pca_path).parent.mkdir(parents=True, exist_ok=True)
np.save(output_pca_path, pca.components_)
print(f"PCA components saved to: {output_pca_path}")

print(f"Explained variance by 128D: {np.sum(pca.explained_variance_ratio_):.4f}")

X_proj = X @ pca.components_.T
print(f"PCA Projected Embedding Mean Norm: {np.mean(np.linalg.norm(X_proj, axis=1)):.4f}")
print(f"Original CLS Embedding Mean Norm: {np.mean(np.linalg.norm(X, axis=1)):.4f}")
