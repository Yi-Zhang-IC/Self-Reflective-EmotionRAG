import os
import json
import torch
import numpy as np
from typing import List, Tuple
from pathlib import Path
from EmotionRAG.get_embeddings import get_semantic_embedding, get_emotion_vector

# ============================
# Device Setup
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# Cosine Distance (Torch)
# ============================

def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    if torch.norm(a) == 0 or torch.norm(b) == 0:
        return 1.0
    sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return 1.0 - sim


# ============================
# Combination Logic
# ============================

def combine_strategy(context_dists, emotion_dists, method: str) -> List[int]:
    context_dists = np.array(context_dists)
    emotion_dists = np.array(emotion_dists)

    if method == "OriginalRAG":
        indices = np.argsort(context_dists)[:10]
    elif method == "C-A":
        combined = context_dists + emotion_dists
        indices = np.argsort(combined)[:10]
    elif method == "C-M":
        combined = context_dists * emotion_dists
        indices = np.argsort(combined)[:10]
    elif method == "S-C":
        top_ctx = np.argsort(context_dists)[:20]
        filtered_emo = emotion_dists[top_ctx]
        indices = top_ctx[np.argsort(filtered_emo)[:10]]
    elif method == "S-S":
        top_emo = np.argsort(emotion_dists)[:20]
        filtered_ctx = context_dists[top_emo]
        indices = top_emo[np.argsort(filtered_ctx)[:10]]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return indices.tolist()

# ============================
# Retrieval Method Definition
# ============================

def retrieval(query_text: str, character: str, database_path: str, method: str = "C-A") -> Tuple[List[str], List[float]]:
    char_path = Path(database_path) / character
    id_map_path = char_path / "id_map.json"
    semantic_pt_path = char_path / "embeddings.pt"
    emotion_jsonl_path = char_path / "gpt_emotion_embeddings.jsonl"

    # Load id_map (string keys)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    mem_ids = list(id_map.keys())

    # Load semantic embeddings
    semantic_tensor = torch.load(semantic_pt_path).to(device)
    if len(mem_ids) != semantic_tensor.shape[0]:
        raise ValueError(f"id_map size ({len(mem_ids)}) ≠ semantic_tensor size ({semantic_tensor.shape[0]})")

    # Load emotion embeddings from jsonl
    emotion_db = {}
    with open(emotion_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            mem_id = record["id"]
            vector = torch.tensor(record["embedding"], dtype=torch.float32, device=device)
            emotion_db[mem_id] = vector

    # Validate all ids exist in emotion_db
    missing_ids = [mem_id for mem_id in mem_ids if mem_id not in emotion_db]
    if missing_ids:
        raise KeyError(f"Missing emotion vectors for mem_ids: {missing_ids[:5]}{' (more omitted)' if len(missing_ids) > 5 else ''}")

    # Get query embeddings
    query_sem = get_semantic_embedding(query_text, normalize=True).to(device)
    query_emo_vec, _ = get_emotion_vector(query_text)
    query_emo = torch.tensor(query_emo_vec, dtype=torch.float32, device=device)

    # Compute distances
    context_distances = []
    emotion_distances = []

    for i, mem_id in enumerate(mem_ids):
        sem_vec = semantic_tensor[i]
        emo_vec = emotion_db[mem_id]

        sem_dist = torch.norm(query_sem - sem_vec, p=2).item()
        emo_dist = cosine_distance(query_emo, emo_vec)

        context_distances.append(sem_dist)
        emotion_distances.append(emo_dist)

    # Combine and select top memories
    indices = combine_strategy(context_distances, emotion_distances, method)

    nearest_memories = [id_map[mem_ids[i]]["text"] for i in indices]
    nearest_scores = [context_distances[i] for i in indices]

    return nearest_memories, nearest_scores