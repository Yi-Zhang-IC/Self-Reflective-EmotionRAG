import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from SR_EmotionRAG.utils import on_trace_update, on_memory_update, extract_json_from_response
from SR_EmotionRAG.generation import openai_generator, load_pipeline
from SR_EmotionRAG.pipeline import run_full_roleplay_pipeline
from SR_EmotionRAG.stage1_prompt import build_stage_1_prompt
from SR_EmotionRAG.stage2_prompt import build_stage_2_prompt
from SR_EmotionRAG.memory_retrieval import retrieve_top_k_memories, retrieve_top_k_hybrid_memories, retrieve_top_k_emotional_memories
# Load the model and generator function
retrievers = {
    "semantic": retrieve_top_k_memories,
    "emotional": retrieve_top_k_emotional_memories,
    "hybrid": retrieve_top_k_hybrid_memories
}

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
json_path = ROOT_DIR / "BFI.json"

# Load question JSON
with open(json_path, "r", encoding="utf-8") as f:
    question_data = json.load(f)

questions = question_data["questions"]
question_ids = sorted(map(int, questions.keys()))

for backend_name in ["llama3", "qwen", "deepseek"]:
    print(f"\n=== Using Generation Backend: {backend_name} ===")
    generate_fn = load_pipeline(generation_backend=backend_name)

    for character in character_list:
        for trial in range(1, 4):  # Run 3 times per character
            output_dir = ROOT_DIR / "evaluation" / "character_response" / "BFI" / "myRAG" / f"{character}" / backend_name
            response_path = output_dir / f"{character}_{trial}_BFI.json"
            trace_path = output_dir / f"{character}_{trial}_BFI_detailed_logs.json"

            print(f"\n=== Running Trial {trial} for Character: {character} using {backend_name} ===")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Resume logic
            if response_path.exists():
                with open(response_path, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
                    responses = saved_data.get("responses", [])
                    answered_ids = {item["id"] for item in responses}
                    print(f"[Resume] Loaded {len(responses)} previous responses.")
            else:
                responses = []
                answered_ids = set()

            if trace_path.exists():
                with open(trace_path, "r", encoding="utf-8") as f:
                    detailed_logs = json.load(f)
                    print(f"[Resume] Loaded {len(detailed_logs)} previous logs.")
            else:
                detailed_logs = []

            for qid in question_ids:
                if qid in answered_ids:
                    print(f"[SKIP] Question {qid} already answered. Skipping.")
                    continue

                q = questions[str(qid)]
                question_text = q["rewritten_en"] if language == "en" else q["rewritten_zh"]

                print(f"\n=== Asking Question {qid}: {question_text} ===")

                try:
                    roleplay_prompt, character_response, reasoning_trace, retrieval_counts = run_full_roleplay_pipeline(
                        max_steps=2,
                        character=character,
                        user_query=question_text,
                        stage1_prompt_fn=build_stage_1_prompt,
                        stage2_prompt_fn=build_stage_2_prompt,
                        llm_generator=openai_generator,
                        retrieve_fn_map=retrievers,
                        generate_fn=generate_fn,
                        on_trace_update=on_trace_update,
                        on_memory_update=on_memory_update
                    )
                except Exception as e:
                    print(f"[ERROR] Failed on Q{qid} for {character}: {e}")
                    continue  # Optionally: raise to stop on failure

                print("Prompt for generation:\n", roleplay_prompt)
                print("Character Response:\n", character_response)

                responses.append({
                    "id": qid,
                    "question": q["rewritten_en"] if language == "en" else q["origin_zh"],
                    "response": character_response.strip(),
                    "dimension": q["dimension"]
                })

                detailed_logs.append({
                    "id": qid,
                    "question": q["rewritten_en"] if language == "en" else q["origin_zh"],
                    "roleplay_prompt": roleplay_prompt,
                    "character_response": character_response.strip(),
                    "reasoning_trace": reasoning_trace,
                    "retrieval_counts": retrieval_counts
                })

                # Save intermediate output after each question
                partial_output = {
                    "character": character.replace('_', ' ').title(),
                    "experimenter": experimenter,
                    "language": language,
                    "backend": backend_name,
                    "responses": responses
                }

                with open(response_path, "w", encoding="utf-8") as f:
                    json.dump(partial_output, f, ensure_ascii=False, indent=2)

                with open(trace_path, "w", encoding="utf-8") as f:
                    json.dump(detailed_logs, f, ensure_ascii=False, indent=2)

                print(f"Intermediate save after Q{qid} for {character} (Trial {trial})")

            print(f"\nTrial {trial} responses saved to {response_path}")
            print(f"Trial {trial} diagnostic logs saved to {trace_path}")
