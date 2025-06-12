import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, CrossEncoder

# =======================
# Path Configs
# =======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_SEMANTIC_MODEL_PATH = str(PROJECT_ROOT / "models" / "bge-base-en-v1.5")
RERANKER_MODEL_PATH = str(PROJECT_ROOT / "models" / "bge-reranker-base")
DATABASE_PATH = PROJECT_ROOT / "database"
TOKENIZER_PATH = str(PROJECT_ROOT / "outputs" / "goemotions_notransfer" / "best_eval_f1_ckpt")
PROJECTION_HEAD_PATH = str(PROJECT_ROOT / "outputs" / "pca" / "pca_components_128.npy")

# =======================
# Device Setup
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# Load Tokenizer Once
# =======================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# =======================
# Embedding + Reranker Models
# =======================
EMBED_MODEL = SentenceTransformer(LOCAL_SEMANTIC_MODEL_PATH)
RERANKER = CrossEncoder(RERANKER_MODEL_PATH)

# =======================
# Projection Model
# =======================
class PCAProjector(nn.Module):
    def __init__(self, pca_components: np.ndarray):  # shape: (128, 768)
        super().__init__()
        self.proj = nn.Linear(768, pca_components.shape[0], bias=False)
        self.proj.weight.data = torch.tensor(pca_components, dtype=torch.float32)
        self.proj.weight.requires_grad = False

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)

class EmotionEmbeddingModel(nn.Module):
    def __init__(self, encoder_path: str, projector: nn.Module, freeze_encoder=True, dropout_rate=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)
        self.dropout = nn.Dropout(dropout_rate)
        self.projector = projector

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("Encoder frozen.")

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = self.dropout(output.last_hidden_state[:, 0])
        return self.projector(cls_embed)

# Load frozen model
model = EmotionEmbeddingModel(
    dropout_rate=0.3,
    projection_dim=128,
    encoder_path=TOKENIZER_PATH,
    projection_head_path=PROJECTION_HEAD_PATH,
    freeze_encoder=True,
    use_projection_head=False
).to(device)
model.eval()

# =======================
# Retrieval Functions
# =======================
@torch.no_grad()
def embed_query_emotionally(query: str):
    encoded = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    return model(encoded["input_ids"], encoded["attention_mask"]).squeeze(0)

def retrieve_top_k_memories(character: str, query: str, k: int = 10, rerank_top: int = 3, rerank_score_threshold: float = 0.0):
    char_dir = DATABASE_PATH / character
    emb_path = char_dir / "embeddings.pt"
    id_map_path = char_dir / "id_map.json"
    assert emb_path.exists(), f"Missing {emb_path}"
    assert id_map_path.exists(), f"Missing {id_map_path}"

    memory_embeddings = torch.load(emb_path)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    query_emb = EMBED_MODEL.encode(f"query: {query}", convert_to_tensor=True)
    similarities = F.cosine_similarity(query_emb, memory_embeddings)
    top_indices = torch.topk(similarities, k=k).indices.tolist()

    dense_top = [{
        "index": idx,
        "text": id_map[str(idx)]["text"],
        "source_paragraph_index": id_map[str(idx)]["source_paragraph_index"],
        "dense_score": round(similarities[idx].item(), 4)
    } for idx in top_indices]

    reranker_inputs = [(query, item["text"]) for item in dense_top]
    reranker_scores = RERANKER.predict(reranker_inputs)
    for i, score in enumerate(reranker_scores):
        dense_top[i]["rerank_score"] = round(score, 4)

    filtered = [m for m in dense_top if m["rerank_score"] >= rerank_score_threshold]
    top_reranked = sorted(filtered, key=lambda x: x["rerank_score"], reverse=True)[:rerank_top]
    return sorted(top_reranked, key=lambda x: x["source_paragraph_index"])

def retrieve_top_k_emotional_memories(character: str, query: str, k: int = 3, sort_by_time=True):
    char_dir = DATABASE_PATH / character
    emb_path = char_dir / "emotion_embeddings.pt"
    id_map_path = char_dir / "id_map.json"
    assert emb_path.exists(), f"Missing {emb_path}"
    assert id_map_path.exists(), f"Missing {id_map_path}"

    memory_embeddings = torch.load(emb_path).to(device)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    query_embedding = embed_query_emotionally(query)
    sims = F.cosine_similarity(query_embedding.unsqueeze(0), memory_embeddings)
    top_indices = torch.topk(sims, k=k).indices.tolist()

    results = [{
        "index": idx,
        "text": id_map[str(idx)]["text"],
        "source_paragraph_index": id_map[str(idx)]["source_paragraph_index"],
        "score": round(sims[idx].item(), 4)
    } for idx in top_indices]

    return sorted(results, key=lambda x: x["source_paragraph_index"]) if sort_by_time else results

def retrieve_top_k_hybrid_memories(character: str, query: str, semantic_top_k=10, rerank_top_k=6, emotion_top_k=3, sort_by_time=True):
    dense_results = retrieve_top_k_memories(character, query, k=semantic_top_k, rerank_top=rerank_top_k)
    if not dense_results:
        return []

    emotional_query_vector = embed_query_emotionally(query)
    char_dir = DATABASE_PATH / character
    emotion_emb_path = char_dir / "emotion_embeddings.pt"
    memory_emotions = torch.load(emotion_emb_path).to(device)

    for result in dense_results:
        idx = result["index"]
        emotion_vec = memory_emotions[idx]
        sim = F.cosine_similarity(emotional_query_vector, emotion_vec, dim=0).item()
        result["emotion_score"] = round(sim, 4)

    top_emotion_results = sorted(dense_results, key=lambda r: r["emotion_score"], reverse=True)[:emotion_top_k]
    return sorted(top_emotion_results, key=lambda r: r["source_paragraph_index"]) if sort_by_time else top_emotion_results

def retrieve_top_k_hybrid_memories_emotion_first(character: str, query: str, emotion_top_k=10, semantic_top_k=3, sort_by_time=True):
    char_dir = DATABASE_PATH / character
    emotion_emb_path = char_dir / "emotion_embeddings.pt"
    semantic_emb_path = char_dir / "embeddings.pt"
    id_map_path = char_dir / "id_map.json"

    assert emotion_emb_path.exists()
    assert semantic_emb_path.exists()
    assert id_map_path.exists()

    emotion_embeddings = torch.load(emotion_emb_path).to(device)
    semantic_embeddings = torch.load(semantic_emb_path).to(device)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    # Step 1: Embed emotional query
    emo_query_emb = embed_query_emotionally(query)
    emo_sim = F.cosine_similarity(emo_query_emb.unsqueeze(0), emotion_embeddings)  # [N]
    top_emo_indices = torch.topk(emo_sim, k=emotion_top_k).indices.tolist()

    # Step 2: Within emotion-retrieved candidates, re-rank using semantic similarity
    sem_query_emb = EMBED_MODEL.encode(f"query: {query}", convert_to_tensor=True).to(device)
    final_results = []
    for idx in top_emo_indices:
        sem_sim = F.cosine_similarity(sem_query_emb, semantic_embeddings[idx], dim=0).item()
        final_results.append({
            "index": idx,
            "text": id_map[str(idx)]["text"],
            "source_paragraph_index": id_map[str(idx)]["source_paragraph_index"],
            "emotion_score": round(emo_sim[idx].item(), 4),
            "semantic_score": round(sem_sim, 4)
        })

    top_results = sorted(final_results, key=lambda r: r["semantic_score"], reverse=True)[:semantic_top_k]
    return sorted(top_results, key=lambda r: r["source_paragraph_index"]) if sort_by_time else top_results

def retrieve_top_k_hybrid_memories_additive_combination(character: str, query: str, top_k: int = 3, sort_by_time: bool = True):
    char_dir = DATABASE_PATH / character
    sem_emb_path = char_dir / "embeddings.pt"
    emo_emb_path = char_dir / "emotion_embeddings.pt"
    id_map_path = char_dir / "id_map.json"

    assert sem_emb_path.exists(), f"Missing {sem_emb_path}"
    assert emo_emb_path.exists(), f"Missing {emo_emb_path}"
    assert id_map_path.exists(), f"Missing {id_map_path}"

    semantic_embeddings = torch.load(sem_emb_path).to(device)
    emotional_embeddings = torch.load(emo_emb_path).to(device)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    query_sem_emb = EMBED_MODEL.encode(f"query: {query}", convert_to_tensor=True).to(device)
    query_emo_emb = embed_query_emotionally(query)

    sem_sim = F.cosine_similarity(query_sem_emb, semantic_embeddings)
    emo_sim = F.cosine_similarity(query_emo_emb.unsqueeze(0), emotional_embeddings)

    combined_score = sem_sim + emo_sim
    top_indices = torch.topk(combined_score, k=top_k).indices.tolist()

    results = [{
        "index": idx,
        "text": id_map[str(idx)]["text"],
        "source_paragraph_index": id_map[str(idx)]["source_paragraph_index"],
        "semantic_score": round(sem_sim[idx].item(), 4),
        "emotion_score": round(emo_sim[idx].item(), 4),
        "combined_score": round(combined_score[idx].item(), 4)
    } for idx in top_indices]

    return sorted(results, key=lambda r: r["source_paragraph_index"]) if sort_by_time else results

def retrieve_top_k_hybrid_memories_multiplicative_combination(character: str, query: str, top_k: int = 3, sort_by_time: bool = True):
    char_dir = DATABASE_PATH / character
    sem_emb_path = char_dir / "embeddings.pt"
    emo_emb_path = char_dir / "emotion_embeddings.pt"
    id_map_path = char_dir / "id_map.json"

    assert sem_emb_path.exists(), f"Missing {sem_emb_path}"
    assert emo_emb_path.exists(), f"Missing {emo_emb_path}"
    assert id_map_path.exists(), f"Missing {id_map_path}"

    semantic_embeddings = torch.load(sem_emb_path).to(device)
    emotional_embeddings = torch.load(emo_emb_path).to(device)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    query_sem_emb = EMBED_MODEL.encode(f"query: {query}", convert_to_tensor=True).to(device)
    query_emo_emb = embed_query_emotionally(query)

    sem_sim = F.cosine_similarity(query_sem_emb, semantic_embeddings)
    emo_sim = F.cosine_similarity(query_emo_emb.unsqueeze(0), emotional_embeddings)

    combined_score = sem_sim * emo_sim
    top_indices = torch.topk(combined_score, k=top_k).indices.tolist()

    results = [{
        "index": idx,
        "text": id_map[str(idx)]["text"],
        "source_paragraph_index": id_map[str(idx)]["source_paragraph_index"],
        "semantic_score": round(sem_sim[idx].item(), 4),
        "emotion_score": round(emo_sim[idx].item(), 4),
        "combined_score": round(combined_score[idx].item(), 4)
    } for idx in top_indices]

    return sorted(results, key=lambda r: r["source_paragraph_index"]) if sort_by_time else results
