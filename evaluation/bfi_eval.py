import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import InferenceClient
import time
import random
import json
from collections import defaultdict

# === OpenAI API Setup ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key, timeout=60)


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


def split_list(input_list, n=4):
    if len(input_list) < 2 * (n-1):
        return [input_list]

    result = [input_list[i:i+n] for i in range(0, len(input_list), n)]
    
    # If last list is too short, balance it out
    num_to_pop = n - 1 - len(result[-1])
    for i in range(num_to_pop):
        result[-1].append(result[i].pop())
        
    return result

def group_responses_by_dimension(response_path):
    with open(response_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    responses_by_dim = defaultdict(list)

    for item in data["responses"]:
        dimension = item["dimension"]
        question = item["question"].strip()
        response = item["response"].strip()
        responses_by_dim[dimension].append((question, response))

    return responses_by_dim

def prepare_batches_per_dimension(response_path, batch_size=4):
    grouped = group_responses_by_dimension(response_path)
    batched = {}

    for dim, qr_pairs in grouped.items():
        batched[dim] = split_list(qr_pairs, n=batch_size)

    return batched

# Load BFI schema once
with open("BFI.json", "r", encoding="utf-8") as f:
    bfi_schema = json.load(f)

bfi_dim_desc = bfi_schema["prompts"]["dim_desc"]
bfi_score_range = bfi_schema["range"]  # Expected to be [1, 3, 5] or similar

def generate_bfi_prompt_auto(
    dimension_name,
    qr_batch,
    language="English",
    experimenter_name="Interviewer",
    character_name="the participant"
):
    scale_name = bfi_schema["name"]
    dimension_description = bfi_dim_desc[dimension_name]
    lowest_score, highest_score = bfi_score_range
    middle_score = (lowest_score + highest_score) / 2

    qr_text = "\n".join([f"Q: {q}\nA: {r}" for q, r in qr_batch])

    prompt = f"""
You are an expert in Psychometrics, especially {scale_name}. I am conducting the {scale_name} test on someone. 
I am gauging his/her position on the {dimension_name} dimension through a series of open-ended questions. 
For clarity, here’s some background on this particular dimension:
===
{dimension_description}
===
My name is {experimenter_name}. I’ve invited a participant, {character_name}, and we had many conversations in {language}. 
Below is a batch of question-and-response pairs from our interview:
===
{qr_text}
===
Please help me assess {character_name}’s score within the {dimension_name} dimension of {scale_name}.
You should provide the score of {character_name} in terms of {dimension_name}, which is a number between {lowest_score} and {highest_score}. 
{lowest_score} denotes ‘not {dimension_name} at all’, {middle_score} denotes ‘neutral’, and {highest_score} denotes ‘strongly {dimension_name}’. 
Other numbers in this range represent different degrees of ‘{dimension_name}’. 
Please output in the following json format:
=== 
{{ 
  "analysis": <your analysis based on the conversations>, 
  "result": <your score> 
}}
"""
    return prompt.strip()

import os
import json
from pathlib import Path
from statistics import mean
import time

def safe_parse_bfi_output(output):
    try:
        parsed = json.loads(output)
        score = float(parsed["result"])
        return parsed["analysis"], score
    except Exception as e:
        print("Failed to parse BFI LLM output. Raw output:")
        print(output[:300] + "...")
        raise e

import os
import json
from pathlib import Path
from statistics import mean
import time

def safe_parse_bfi_output(output):
    try:
        parsed = json.loads(output)
        score = float(parsed["result"])
        return parsed["analysis"], score
    except Exception as e:
        print("Failed to parse BFI LLM output. Raw output:")
        print(output[:300] + "...")
        raise e

def evaluate_character_bfi_er_batch(character_backend_path, model="gpt-4.1", max_retries=3):
    character_name = Path(character_backend_path).parts[-2]
    backend_name = Path(character_backend_path).parts[-1]
    all_dim_scores = {}

    response_files = [
        f for f in os.listdir(character_backend_path)
        if f.endswith(".json") and not (
            f.endswith("_summary.json")
            or f.endswith("_eval.json")
            or f.endswith("_logs.json")
        )
    ]

    if len(response_files) != 3:
        raise RuntimeError(f"{character_backend_path} contains {len(response_files)} response files (expected 3)")

    for response_file in response_files:
        full_path = os.path.join(character_backend_path, response_file)
        out_resp_path = os.path.join(character_backend_path, response_file.replace(".json", "_er_eval.json"))

        if os.path.exists(out_resp_path):
            print(f"Skipping {response_file} (already evaluated)")
            continue

        batches_by_dim = prepare_batches_per_dimension(full_path)
        per_response_output = {
            "character": character_name,
            "backend": backend_name,
            "response_file": response_file,
            "dimensions": {}
        }

        try:
            for dim, batches in batches_by_dim.items():
                batch_scores = []

                for batch in batches:
                    prompt = generate_bfi_prompt_auto(
                        dimension_name=dim,
                        qr_batch=batch,
                        character_name="the participant",
                        experimenter_name="Interviewer"
                    )

                    for attempt in range(max_retries + 1):
                        try:
                            output = openai_generator(prompt, model=model)[0]["generated_text"]
                            analysis, score = safe_parse_bfi_output(output)
                            batch_scores.append(score)
                            break
                        except Exception as e:
                            print(f"Attempt {attempt+1}/{max_retries+1} failed for {response_file}, {dim}: {e}")
                            time.sleep(2)
                            if attempt == max_retries:
                                raise RuntimeError(f"All retries failed for {response_file}, {dim}")

                avg_score = round(mean(batch_scores), 2)
                per_response_output["dimensions"][dim] = {
                    "batch_scores": batch_scores,
                    "average_score": avg_score
                }

                if dim not in all_dim_scores:
                    all_dim_scores[dim] = []
                all_dim_scores[dim].append(avg_score)

            required_dims = set(bfi_schema["prompts"]["dim_desc"].keys())
            missing_dims = required_dims - set(per_response_output["dimensions"].keys())
            if missing_dims:
                raise RuntimeError(f"Incomplete evaluation for {response_file}. Missing dimensions: {missing_dims}")

            with open(out_resp_path, "w", encoding="utf-8") as f:
                json.dump(per_response_output, f, indent=2)
            print(f"Saved: {out_resp_path}")

        except Exception as e:
            if os.path.exists(out_resp_path):
                os.remove(out_resp_path)
            print(f"Evaluation failed for {response_file}: {e}")
            raise e

    # Final summary
    summary = {
        "character": character_name,
        "backend": backend_name,
        "averages": {}
    }

    for dim, scores in all_dim_scores.items():
        avg_score = round(mean(scores), 2)
        summary["averages"][dim] = avg_score

    out_summary_path = os.path.join(character_backend_path, f"{character_name}_{backend_name}_bfi_er_eval_summary.json")
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Final summary written: {out_summary_path}")

def validate_bfi_structure(base_dir="character_response/BFI"):
    rag_types = sorted(os.listdir(base_dir))
    invalid_paths = []

    for rag in rag_types:
        rag_path = os.path.join(base_dir, rag)
        if not os.path.isdir(rag_path):
            continue

        characters = sorted(os.listdir(rag_path))
        for character in characters:
            character_path = os.path.join(rag_path, character)
            if not os.path.isdir(character_path):
                continue

            backends = sorted(os.listdir(character_path))
            for backend in backends:
                if backend.lower() == "mistral":
                    continue

                backend_path = os.path.join(character_path, backend)
                if not os.path.isdir(backend_path):
                    continue

                # Use refined logic to count only true response files
                response_files = [
                    f for f in os.listdir(backend_path)
                    if f.endswith(".json") and not (
                        f.endswith("_summary.json")
                        or f.endswith("_eval.json")
                        or f.endswith("_logs.json")
                    )
                ]

                if len(response_files) != 3:
                    invalid_paths.append((backend_path, len(response_files)))

    if invalid_paths:
        for path, count in invalid_paths:
            print(f"Invalid path: {path} — contains {count} response files (expected 3)")
        raise RuntimeError(f"{len(invalid_paths)} backend paths failed validation. Aborting.")

    print("All character/backend paths passed validation.")

def run_bfi_evaluation_all(base_dir="character_response/BFI"):
    rag_types = sorted(os.listdir(base_dir))

    for rag in rag_types:
        rag_path = os.path.join(base_dir, rag)
        if not os.path.isdir(rag_path):
            continue

        characters = sorted(os.listdir(rag_path))
        for character in characters:
            character_path = os.path.join(rag_path, character)
            if not os.path.isdir(character_path):
                continue

            backends = sorted(os.listdir(character_path))
            for backend in backends:
                if backend.lower() == "mistral":
                    continue

                backend_path = os.path.join(character_path, backend)
                if not os.path.isdir(backend_path):
                    continue

                print(f"\n=== Evaluating: {rag}/{character}/{backend} ===")
                evaluate_character_bfi_er_batch(backend_path)

# First validate
# validate_bfi_structure()

# Then run evaluations only if validation passed
run_bfi_evaluation_all()













