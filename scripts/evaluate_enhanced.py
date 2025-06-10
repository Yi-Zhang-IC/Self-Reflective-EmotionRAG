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
from transformers import AutoModelForSequenceClassification
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
        encoder_path: str,
        use_mlp_output: bool = False,
        dropout_rate: float = 0.3,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.use_mlp_output = use_mlp_output
        self.dropout = nn.Dropout(dropout_rate)

        # Load full model (encoder + MLP head)
        self.model = AutoModelForSequenceClassification.from_pretrained(encoder_path)

        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.model.roberta.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        if self.use_mlp_output:
            logits = self.model.classifier(outputs.last_hidden_state)  # ✅ Full hidden state
            return F.normalize(logits, p=2, dim=1)  # 28D normalized embedding
        else:
            cls_embed = self.dropout(outputs.last_hidden_state[:, 0])
            return F.normalize(cls_embed, p=2, dim=1)  # 768D CLS embedding

# -----------------------------
# 3. Evaluation
# -----------------------------
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
        labels = batch["label"].to(device)

        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)

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

# -----------------------------
# 4. TSNE Plot
# -----------------------------
def plot_top_combinations_tsne(embeddings, label_vectors, emotion_names, save_dir="combo_tsne_plots", top_k=12):
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
    neutral_index = emotion_names.index("neutral")
    combo_counter = Counter([
        combo for combo in label_tuples
        if len(combo) == 2 and neutral_index not in combo
    ])
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
    projection_head_path: str = None,
    freeze_encoder: bool = True,
    encoder_path: str = "models/roberta-base-go_emotions",
    test_jsonl_path: str = "data/augmented_go_emotion/test.jsonl",
    tokenizer_path: str = "models/roberta-base-go_emotions",
    tsne_save_path: str = "embedding_plot_named",
    use_mlp_output: bool = True,
    eval_ks=[1, 3, 6]
):
    import torch
    import json
    import logging
    from pathlib import Path
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = EmotionDataset(test_jsonl_path, tokenizer)

    # ---------- Validation ----------
    if encoder_path is None:
        raise ValueError("`encoder_path` must be provided.")

    # ---------- Model Setup ----------
    model = EmotionEmbeddingModel(
        encoder_path=encoder_path,
        use_mlp_output= use_mlp_output,
        freeze_encoder= freeze_encoder,
    )

    model.to(device)

    # ---------- Logging Setup ----------
    log_path = Path(tsne_save_path) / "evaluation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("embedding-eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # clear existing handlers
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info("▶️ Running evaluation...")
    logger.info(f"📦 Encoder path: {encoder_path}")
    if projection_head_path:
        logger.info(f"📎 Projection head path: {projection_head_path}")
    logger.info(f"🧊 Encoder frozen: {freeze_encoder}")

    # ---------- Quantitative Evaluation ----------
    metrics_by_k = evaluate_embeddings_at_ks(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=128,
        ks=eval_ks,
        num_classes=28
    )

    logger.info("📊 Embedding Space Evaluation Metrics:")
    for k, metric_set in metrics_by_k.items():
        logger.info(f"--- k={k} ---")
        for metric, value in metric_set.items():
            logger.info(f"{metric:>20s}: {value:.4f}")

    # Save metrics as JSON
    def sanitize_metrics(metrics):
        def safe(val):
            if isinstance(val, torch.Tensor):
                return val.item() if val.numel() == 1 else val.tolist()
            if isinstance(val, (float, int, str)):
                return val
            if hasattr(val, "tolist"):
                return val.tolist()
            return float(val)  # fallback
        return {
            k: {kk: safe(vv) for kk, vv in v.items()}
            for k, v in metrics.items()
        }

    serializable_metrics = sanitize_metrics(metrics_by_k)
    with open(Path(tsne_save_path) / "evaluation_metrics.json", "w") as f:
        json.dump(serializable_metrics, f, indent=2)

    # ---------- t-SNE Preparation ----------
    all_embeddings = []
    all_labels = []

    loader = DataLoader(dataset, batch_size=64)
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            emb = model(input_ids=input_ids, attention_mask=attention_mask)
        all_embeddings.append(emb.cpu())
        all_labels.extend(batch["label"])

    # ---------- t-SNE Plot ----------
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
        save_dir=tsne_save_path,
        top_k=8
    )

    logger.info("✅ Evaluation complete.")
    return metrics_by_k  # Optional: return for in-notebook inspection

run_evaluation(
    encoder_path="outputs/goemotions_transfer2/best_eval_f1_ckpt",
    use_mlp_output=False,
    freeze_encoder=True,
    tsne_save_path="evaluation/embedding_tsne/goemotions_transfer2/best_eval_f1",
    test_jsonl_path="data/augmented_go_emotion/test.jsonl",
    tokenizer_path="outputs/goemotions_transfer2/best_eval_f1_ckpt"
)
