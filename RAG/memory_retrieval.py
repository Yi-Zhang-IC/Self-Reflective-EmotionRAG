import torch
import torch.nn as nn
import torch.nn.functional as F
import json
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
TOKENIZER_PATH = str(PROJECT_ROOT / "models" / "roberta-base-go_emotions")
PROJECTION_HEAD_PATH = str(PROJECT_ROOT / "outputs" / "stage_one_trainning_va" / "best_projection_head.pt")

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
    def __init__(self, dropout_rate=0.3, projection_dim=128, model_path=None, encoder_path=None,
                 projection_head_path=None, freeze_encoder=False):
        super().__init__()
        assert encoder_path is not None, "You must specify encoder_path"

        self.encoder = AutoModel.from_pretrained(encoder_path)
        hidden_size = self.encoder.config.hidden_size
        self.projection_head = ProjectionHead(hidden_size, 256, projection_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if projection_head_path:
                self.projection_head.load_state_dict(torch.load(projection_head_path, map_location=device))
        else:
            assert model_path is not None, "model_path required for non-frozen encoder"
            self.load_state_dict(torch.load(model_path, map_location=device))

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = self.dropout(output.last_hidden_state[:, 0])
        return self.projection_head(cls_embed)

# Load frozen model
model = EmotionEmbeddingModel(
    projection_head_path=PROJECTION_HEAD_PATH,
    encoder_path=TOKENIZER_PATH,
    freeze_encoder=True
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
