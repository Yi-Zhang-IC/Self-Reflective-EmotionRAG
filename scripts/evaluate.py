import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path
from typing import Union
import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from collections import Counter, defaultdict
from pathlib import Path
# -----------------------------
# 1. Dataset
# -----------------------------
class EmotionDataset(Dataset):
    def __init__(self, path: Union[str, Path], tokenizer, num_classes=28, max_length=128):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)

        self.tokenizer = tokenizer
        self.num_classes = num_classes
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
        label = item["labels"]
        label_vector = torch.zeros(self.num_classes)
        for l in label:
            label_vector[l] = 1
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": label_vector
        }

    def __len__(self):
        return len(self.samples)

# -----------------------------
# 2. Model
# -----------------------------
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
        return F.normalize(x, p=2, dim=1)

class EmotionEmbeddingModel(nn.Module):
    def __init__(
        self,
        dropout_rate=0.3,
        projection_dim=128,
        model_path=None,
        encoder_path=Path("models/roberta-base-go_emotions"),
        projection_head_path=None,
        freeze_encoder=False
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

        # Always load encoder from pretrained dir
        self.encoder = AutoModel.from_pretrained(encoder_path)
        hidden_size = self.encoder.config.hidden_size
        self.projection_head = ProjectionHead(hidden_size, 256, projection_dim, dropout_rate)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if projection_head_path is not None:
                proj_state_dict = torch.load(projection_head_path)  # Load on GPU by default
                self.projection_head.load_state_dict(proj_state_dict)
        else:
            assert model_path is not None, "Must provide `modle_path` when freeze_encoder is False"
            state_dict = torch.load(model_path)  # .pt or .bin file containing full model weights
            self.load_state_dict(state_dict)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = self.dropout(output.last_hidden_state[:, 0])
        return self.projection_head(cls_embed)


# -----------------------------
# 3. Evaluation
# -----------------------------
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

# -----------------------------
# 4. TSNE Plot
# -----------------------------
def plot_top_combinations_tsne(embeddings, label_vectors, emotion_names, save_dir="combo_tsne_plots", top_k=8):
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    label_tuples = [tuple(np.nonzero(lbl.numpy())[0]) for lbl in label_vectors]
    embedding_array = torch.cat(embeddings, dim=0).cpu().numpy()

    # ---------------------------
    # 0. Full single-label plot
    # ---------------------------
    single_indices = [i for i, lbl in enumerate(label_vectors) if lbl.sum().item() == 1]
    if len(single_indices) > 0:
        single_embeddings = torch.cat(embeddings, dim=0)[single_indices]
        single_labels = [label.argmax().item() for i, label in enumerate(label_vectors) if i in single_indices]

        tsne = TSNE(
            n_components=2, perplexity=50, n_iter=1500,
            learning_rate=300, metric="cosine", init="pca", random_state=42
        )
        reduced = tsne.fit_transform(single_embeddings.numpy())

        label_ids = np.array(single_labels)

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=label_ids, cmap='tab20', alpha=0.6, s=20)

        # Manually ensure all 28 emotions are in the legend
        for class_id in range(len(emotion_names)):
            plt.scatter([], [], c=[plt.cm.tab20(class_id / 28)], label=emotion_names[class_id])

        plt.title("t-SNE of Single-Label Emotion Embeddings")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        fname = f"{save_dir}/tsne_all_single_labels.png"
        plt.savefig(fname)
        plt.close()
        logging.info(f"Saved full single-label t-SNE plot to: {fname}")


    # ---------------------------
    # 1. Combo plots (top-k co-occurrence)
    # ---------------------------
    combo_counter = Counter([combo for combo in label_tuples if len(combo) == 2])
    top_combos = combo_counter.most_common(top_k)

    for (a, b), _ in top_combos:
        indices_single_a = [i for i, l in enumerate(label_tuples) if l == (a,)]
        indices_single_b = [i for i, l in enumerate(label_tuples) if l == (b,)]
        indices_combo = [i for i, l in enumerate(label_tuples) if set(l) == {a, b}]
        indices = indices_single_a + indices_single_b + indices_combo

        if len(indices) < 20:
            continue  # skip tiny groups that make bad plots

        selected_embeddings = embedding_array[indices]
        selected_labels = (
            ["A"] * len(indices_single_a) +
            ["B"] * len(indices_single_b) +
            ["A+B"] * len(indices_combo)
        )

        tsne = TSNE(n_components=2, perplexity=50, n_iter=1500, learning_rate=300,
                    metric="cosine", init="pca", random_state=42)
        reduced = tsne.fit_transform(selected_embeddings)

        plt.figure(figsize=(10, 8))
        for cls, color in zip(["A", "B", "A+B"], ['blue', 'green', 'red']):
            mask = np.array(selected_labels) == cls
            plt.scatter(reduced[mask, 0], reduced[mask, 1], label=cls, alpha=0.6, s=20, c=color)

        title = f"{emotion_names[a]} + {emotion_names[b]}"
        plt.title(f"t-SNE for Combo: {title}")
        plt.legend()
        fname = f"{save_dir}/tsne_{emotion_names[a]}_{emotion_names[b]}.png".replace(" ", "_")
        plt.savefig(fname)
        plt.close()
        print(f"Saved combo plot: {fname}")

# -----------------------------
# 5. Main
# -----------------------------
def run_evaluation(
    model_path: str = None,
    projection_head_path: str = None,
    freeze_encoder: bool = True,
    test_jsonl_path: str = "data/augmented_go_emotion/test.jsonl",
    tokenizer_path: str = "models/roberta-base-go_emotions",
    tsne_save_path: str = "embedding_plot_named"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = EmotionDataset(test_jsonl_path, tokenizer)

    # ---------- Validation ----------
    if (model_path is None) == (projection_head_path is None):
        raise ValueError("Specify either `model_path` (for fine-tuned model) OR `projection_head_path` (for frozen encoder), not both or neither.")

    # ---------- Model Setup ----------
    model = EmotionEmbeddingModel(
        dropout_rate=0.3,
        projection_dim=128,
        model_path=model_path,
        projection_head_path=projection_head_path,
        freeze_encoder=freeze_encoder
    ).to(device)

    # ---------- Evaluation ----------
    macro_prec, class_avg_precisions = evaluate_embeddings_macro_precision(model, dataset, device)
    # create a log file if it doesn't exist
    Path(tsne_save_path).mkdir(exist_ok=True, parents=True)
    with open(f"{tsne_save_path}/evaluation_log.txt", "a") as f:
        f.write(f"Macro Precision: {macro_prec:.4f}\n")
        f.write(f"Class Average Precisions: {class_avg_precisions}\n")

    # ---------- TSNE Visualization ----------
    all_embeddings = []
    all_labels = []
    for batch in DataLoader(dataset, batch_size=64):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            emb = model(input_ids=input_ids, attention_mask=attention_mask)
        all_embeddings.append(emb.cpu())
        all_labels.extend(batch["label"])

    plot_top_combinations_tsne(
        embeddings=all_embeddings,
        label_vectors=all_labels,
        emotion_names=[
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
            "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
            "nervousness", "optimism", "pride", "realization", "relief", "remorse",
            "sadness", "surprise", "neutral"
        ],
        save_dir=tsne_save_path,  # or any output path
        top_k=8
)

run_evaluation(
    model_path="outputs/NT_XENT/best_full_model_train_encoder.pt",
    freeze_encoder=False,
    test_jsonl_path="data/augmented_go_emotion/test.jsonl",
    tokenizer_path="models/roberta-base-go_emotions",
    tsne_save_path="evaluation/embedding_tsne/NT_XENT_unfreeze_encoder" 
)
