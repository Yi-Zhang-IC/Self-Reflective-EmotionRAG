# build_goemotion_va_similarity.py (manual mode)

import pandas as pd
import torch
import os

# Manually downloaded NRC-VAD file path
vad_txt = "scripts/NRC-VAD-Lexicon.txt"

# GoEmotions class names (ensure order matches label index)
goemotion_classes = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# Load NRC-VAD
df = pd.read_csv(vad_txt, sep="\t", names=["word", "valence", "arousal", "dominance"], skiprows=1)
df = df.set_index("word")

# Normalize V-A vectors
vad_embeddings = []
for emotion in goemotion_classes:
    word = emotion.lower()
    if word in df.index:
        v = df.loc[word, ["valence", "arousal"]].values.astype(float)
    else:
        v = [0.5, 0.5]  # fallback if word not found
    vad_embeddings.append(v)

vad_tensor = torch.tensor(vad_embeddings)
vad_norm = torch.nn.functional.normalize(vad_tensor, p=2, dim=1)
similarity = torch.matmul(vad_norm, vad_norm.T)

# Save as CSV
save_path = "data/goemotions_va.csv"
pd.DataFrame(similarity.numpy()).to_csv(save_path, index=False, header=False)
print(f"Saved similarity matrix to {save_path}")
