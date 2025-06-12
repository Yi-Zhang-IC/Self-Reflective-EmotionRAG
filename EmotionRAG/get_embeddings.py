import torch
import torch.nn.functional as F
import json
import time
from tqdm import tqdm
from pathlib import Path
from pathlib import Path
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from EmotionRAG.generation import openai_generator

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


# === Emotion Scoring Prompt ===
system_prompt = """
Task Description: You are an expert in emotion detection. Given a paragraph or a question, your task is to analyze how strongly the it expresses each of the eight basic emotions: joy, acceptance, fear, surprise, sadness, disgust, anger, and anticipation.

Scoring Criteria: For each of the eight emotions, assign a score from 1 to 10 indicating the strength with which the emotion is conveyed in the question. A score of 1 means the emotion is not present or barely detectable, while a score of 10 means the emotion is strongly expressed. Provide a brief explanation for each score based on your analysis.

Output Format: Your output should be a valid Python list of dictionaries, each containing the keys "dim" (the name of the emotion), "score" (an integer from 1 to 10), and "analysis" (a brief justification for the score). The list must contain exactly eight elements, one for each emotion.

[
    {"analysis": <REASON>, "dim": "joy", "score": <SCORE>},
    ...
    {"analysis": <REASON>, "dim": "anticipation", "score": <SCORE>}
]
"""

# === Emotion Extraction Function ===
def get_emotion_vector(text, max_parse_retries=3):
    user_prompt = f"\n{text}"
    for attempt in range(max_parse_retries):
        raw = ""
        try:
            raw = openai_generator(system_prompt=system_prompt, prompt=user_prompt)
            emotion_list = json.loads(raw)
            break
        except Exception as e:
            print(f"[Attempt {attempt + 1}] Parsing failed: {e}")
            print("Raw output:", raw)
            time.sleep(1 + 0.5 * attempt)
    else:
        raise Exception("Failed to parse emotion vector after retries.")

    emotions = ["joy", "acceptance", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
    scores = [None] * 8
    for item in emotion_list:
        if item["dim"] in emotions:
            scores[emotions.index(item["dim"])] = item["score"]
    return scores, raw

# === Semantic Embedding Function ===
def get_semantic_embedding(text: str, normalize: bool = True) -> torch.Tensor:
    embedding = EMBED_MODEL.encode(text, convert_to_tensor=True, device=device)
    if normalize:
        embedding = F.normalize(embedding, p=2, dim=-1)
    return embedding
