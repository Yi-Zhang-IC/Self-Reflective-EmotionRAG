import json
import sys
from pathlib import Path
import traceback    

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from EmotionRAG.get_response import get_response
from EmotionRAG.generation import load_pipeline

# === Configuration ===
character_list = [
    "albus_dumbledore",
    "draco_malfoy",
    "harry_potter",
    "hermione_granger",
    "luna_lovegood",
    "minerva_mcgonagall",
    "ron_weasley",
    "severus_snape"
]

experimenter = "Interviewer"
language = "en"
json_path = ROOT_DIR / "16Personalities.json"
database_path = ROOT_DIR / "database"
retrival_method = "OriginalRAG"

# Load questions
with open(json_path, "r", encoding="utf-8") as f:
    question_data = json.load(f)
questions = question_data["questions"]
question_ids = sorted(map(int, questions.keys()))

# === Run per backend ===
for backend_name in ["llama3", "qwen", "deepseek"]:
    print(f"\n=== Using Generation Backend: {backend_name} ===")
    generate_fn = load_pipeline(generation_backend=backend_name)

    for character in character_list:
        for trial in range(1, 4):
            output_dir = ROOT_DIR / "evaluation" / "character_response" / "MBTI" / "RAG" / f"{character}" / backend_name
            output_dir.mkdir(parents=True, exist_ok=True)

            response_path = output_dir / f"{character}_{trial}_MBTI.json"

            # Resume: load existing data
            if response_path.exists():
                with open(response_path, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
                    responses = saved_data.get("responses", [])
                    answered_ids = {item["id"] for item in responses}
                    print(f"[Resume] Loaded {len(responses)} responses.")
            else:
                responses = []
                answered_ids = set()

            for qid in question_ids:
                if qid in answered_ids:
                    print(f"[SKIP] Question {qid} already answered.")
                    continue

                q = questions[str(qid)]
                question_text = q["rewritten_en"] if language == "en" else q["rewritten_zh"]

                print(f"\n=== Asking Question {qid}: {question_text} ===")

                try:
                    response = get_response(
                        character=character,
                        question=question_text,
                        database_path=str(database_path),
                        method=retrival_method,
                        generate_fn=generate_fn
                    )
                except Exception as e:
                    print(f"[FATAL] Failed Q{qid} for {character} using backend '{backend_name}': {e}")
                    traceback.print_exc()
                    print(f"[FATAL] Halting script. No output saved for this question.")
                    sys.exit(1)

                print("Character Response:\n", response)

                responses.append({
                    "id": qid,
                    "question": q["rewritten_en"] if language == "en" else q["origin_zh"],
                    "response": response.strip(),
                    "dimension": q["dimension"]
                })

                # Save immediately
                output_json = {
                    "character": character.replace('_', ' ').title(),
                    "experimenter": experimenter,
                    "language": language,
                    "backend": backend_name,
                    "responses": responses
                }

                with open(response_path, "w", encoding="utf-8") as f:
                    json.dump(output_json, f, ensure_ascii=False, indent=2)

                print(f"Saved Q{qid} to {response_path}")

            print(f"\nTrial {trial} complete for {character} — file saved at {response_path}")
