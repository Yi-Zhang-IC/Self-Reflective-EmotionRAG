import os
import torch
import json
from tqdm import tqdm
from pathlib import Path
from EmotionRAG.get_embeddings import get_emotion_vector

def load_existing_jsonl(jsonl_path):
    processed_ids = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["id"])
                except json.JSONDecodeError:
                    continue
    return processed_ids

def append_to_jsonl(jsonl_path, record):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def generate_all_emotion_embeddings(database_path="../database"):
    characters = [c for c in os.listdir(database_path) if os.path.isdir(os.path.join(database_path, c))]

    for character in tqdm(characters, desc="Characters"):
        print(f"\n🔍 Processing character: {character}")
        char_dir = os.path.join(database_path, character)
        id_map_path = os.path.join(char_dir, "id_map.json")
        output_pt_path = os.path.join(char_dir, "gpt_emotion_embeddings.pt")
        output_jsonl_path = os.path.join(char_dir, "gpt_emotion_embeddings.jsonl")

        if not os.path.exists(id_map_path):
            print(f"Skipping {character}: id_map.json not found.")
            raise

        with open(id_map_path, "r", encoding="utf-8") as f:
            id_map = json.load(f)

        # Load already processed IDs if resuming
        processed_ids = load_existing_jsonl(output_jsonl_path)

        # Load existing embeddings if resuming
        if os.path.exists(output_pt_path):
            emotion_embeddings = torch.load(output_pt_path)
        else:
            emotion_embeddings = {}

        # Process each memory
        for mem_id, entry in tqdm(id_map.items(), desc=f"  ↪ Memories in {character}"):
            if mem_id in processed_ids:
                continue  # Skip if already done

            text = entry["text"]
            try:
                vector, _ = get_emotion_vector(text)
            except Exception as e:
                print(f"Failed to get emotion vector for memory {mem_id}: {e}")
                raise
            emotion_embeddings[mem_id] = vector

            # Save JSONL line
            record = {
                "id": mem_id,
                "text": text,
                "embedding": vector
            }
            append_to_jsonl(output_jsonl_path, record)

        # Save updated PT file
        torch.save(emotion_embeddings, output_pt_path)
        print(f"Saved: {output_pt_path}")
        print(f"Updated: {output_jsonl_path}")

# === Entry Point ===
project_root = Path(__file__).resolve().parent.parent
database_path = project_root / "database"
generate_all_emotion_embeddings(database_path=str(database_path))
