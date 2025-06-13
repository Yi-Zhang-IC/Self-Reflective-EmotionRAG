import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import InferenceClient
import time
import random

# === OpenAI API Setup ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key, timeout=60)

# === Hugging Face API Setup ===
HF_API_KEY = os.getenv("HF_TOKEN")
hf_client = InferenceClient(api_key=HF_API_KEY, provider="auto")  # can set provider like "novita"

# === Hugging Face Local LLM Generator ===
class HuggingFaceCallFailed(Exception):
    pass

def huggingface_api_generator(
    prompt: str,
    model_name: str = "deepseek-ai/DeepSeek-V3-0324",
    max_tokens: int = 400,
    temperature: float = 0.8,
    top_p: float = 0.92,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    for attempt in range(retries):
        try:
            response = hf_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Attempt {attempt+1}] HF API generation failed: {e}")
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise HuggingFaceCallFailed(f"All {retries} attempts failed. Last error: {e}")


# === OpenAI Generator ===
class OpenAICallFailed(Exception):
    pass

def openai_generator(prompt: str, system_prompt: str = "", model: str = "gpt-4.1-nano",
                     temperature: float = 0.7, top_p=1.0, max_tokens: int = 1000,
                     max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            # ⬅️ wrap string inside list of dicts
            return [{"generated_text": response.choices[0].message.content.strip()}]
        except Exception as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"OpenAI call failed (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(wait)

    raise OpenAICallFailed("OpenAI call failed after multiple retries.")


def load_pipeline(generation_backend):
    """
    Returns a function generate_fn(prompt) that uses the selected backend model to generate text.
    """
    if generation_backend in {"deepseek", "mistral", "qwen", "llama3"}:
        hf_model_map = {
            "deepseek": "deepseek-ai/DeepSeek-V3-0324",
            "qwen": "Qwen/Qwen2.5-32B-Instruct",
            "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",  # Add llama3 here
        }
        model_name = hf_model_map[generation_backend]
        generate_fn = lambda prompt: huggingface_api_generator(prompt, model_name=model_name)
    
    elif generation_backend == "openai":
        generate_fn = lambda prompt: openai_generator(prompt)
    
    else:
        raise ValueError(f"Unknown backend: {generation_backend}")
    
    return generate_fn
