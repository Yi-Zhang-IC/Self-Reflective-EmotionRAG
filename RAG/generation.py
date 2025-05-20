import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from openai import OpenAI

# === Constants ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

OPENHERMES_MODEL_PATH = MODELS_DIR / "OpenHermes-2.5-Mistral-7B"
MISTRAL_MODEL_PATH = MODELS_DIR / "Mistral-7B-v0.1"
DEEPSEEK_MODEL_PATH = MODELS_DIR / "DeepSeek-R1-Distill-Qwen-7B"

# === OpenAI API Setup ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key, timeout=60)


# === Local Model Loader ===
def load_local_roleplay_model(model_path: Path) -> pipeline:
    """
    Load a local Hugging Face causal LM pipeline from a specified path.
    """
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False,
            local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True
        )
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"Failed to load local model from {model_path}")
        raise e          

# === Local LLM Generator ===
def local_llm_generator(
    local_pipeline,
    prompt: str,
    max_new_tokens: int = 800,
    temperature=0.8,
    top_p=0.92
):
    try:
        result = local_pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            return_full_text=False
        )
        return [{"generated_text": result[0]["generated_text"].strip()}]
    except Exception as e:
        print("Local LLM generation failed:", e)
        return [{"generated_text": ""}]


# === OpenAI Generator ===
def openai_generator(
    prompt: str,
    model: str = "gpt-4.1-nano",
    temperature: float = 0.7,
    max_tokens: int = 1000
):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an intelligent reasoning assistant. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content.strip()
        return [{"generated_text": content}]
    except Exception as e:
        print(f"OpenAI generation failed: {e}")
        return [{"generated_text": ""}]
