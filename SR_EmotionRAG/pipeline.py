from typing import Callable, Tuple, List, Dict
import time
import json
from pathlib import Path

from SR_EmotionRAG.utils import get_reasoning_output

def multistep_rag_loop_two_stage(
    character: str,
    user_query: str,
    stage1_prompt_fn,
    stage2_prompt_fn,
    llm_generator,
    retrieve_fn_map,
    max_steps: int = 3,
    max_retries: int = 5,
    retry_delay: float = 1.0,
    on_trace_update: Callable[[Dict], None] = None,
    on_memory_update: Callable[[Dict], None] = None
):
    obtained_memories = []
    reasoning_trace = []
    seen_indices = set()

    retrieval_counts = {
        "semantic": 0,
        "hybrid": 0
    }

    # Stage 1
    stage1_prompt = stage1_prompt_fn(character, user_query)

    # Retry loop with validation
    stage1_output = None
    for attempt in range(max_retries + 1):
        output = get_reasoning_output(stage1_prompt, llm_generator)

        # Validate structure: list of dicts with required keys
        if (
            isinstance(output, list)
            and all(isinstance(q, dict) and "query" in q and "retrieval_type" in q for q in output)
        ):
            stage1_output = output
            break  # Valid output
        else:
            print(f"[Stage 1 Retry {attempt + 1}] Invalid or malformed output: {output}")
            time.sleep(retry_delay * (2 ** attempt))

    # After retries, if still invalid, raise an error
    if stage1_output is None:
        raise ValueError("Stage 1 failed: Could not obtain valid reasoning output from LLM after retries.")

    # Proceed with validated output
    stage1_queries = stage1_output
    stage1_record = {
        "step": "stage_1",
        "original_query": user_query,
        "planned_queries": stage1_queries
    }
    reasoning_trace.append(stage1_record)

    if on_trace_update:
        try:
            on_trace_update(stage1_record)
        except Exception as e:
            print(f"[Warning] on_trace_update callback failed: {e}")


    for q in stage1_queries:
        q_text = q["query"]
        q_type = q["retrieval_type"]
        if q_type not in retrieve_fn_map:
            continue
        retrieval_counts[q_type] += 1
        try:
            results = retrieve_fn_map[q_type](character, q_text)
        except Exception as e:
            print(f"Retrieval failed: {e}")
            continue
        for item in results:
            para_idx = item.get("source_paragraph_index")
            if para_idx is not None and para_idx not in seen_indices:
                mem = {
                    "source_query": q_text,
                    "text": item["text"],
                    "source_paragraph_index": para_idx
                }
                obtained_memories.append(mem)
                seen_indices.add(para_idx)
                if on_memory_update:
                    on_memory_update(mem)

    # Stage 2
    for step in range(max_steps):
        stage2_prompt = stage2_prompt_fn(character, user_query, obtained_memories)
        for attempt in range(max_retries + 1):
            stage2_output = get_reasoning_output(stage2_prompt, llm_generator)
            if stage2_output is not None:
                break
            time.sleep(retry_delay * (2 ** attempt))
        if stage2_output is None:
            break

        reason = stage2_output.get("reason", "")
        queries = stage2_output.get("queries", [])
        step_record = {
            "step": f"stage_2_{step}",
            "reason": reason,
            "queries": queries
        }
        reasoning_trace.append(step_record)
        if on_trace_update:
            on_trace_update(step_record)

        if not queries:
            break

        for q in queries:
            q_text = q["query"]
            q_type = q["retrieval_type"]
            if q_type not in retrieve_fn_map:
                continue
            retrieval_counts[q_type] += 1
            try:
                results = retrieve_fn_map[q_type](character, q_text)
            except Exception as e:
                print(f"Retrieval failed: {e}")
                continue
            for item in results:
                para_idx = item.get("source_paragraph_index")
                if para_idx is not None and para_idx not in seen_indices:
                    mem = {
                        "source_query": q_text,
                        "text": item["text"],
                        "source_paragraph_index": para_idx
                    }
                    obtained_memories.append(mem)
                    seen_indices.add(para_idx)
                    if on_memory_update:
                        on_memory_update(mem)

    obtained_memories.sort(key=lambda m: m["source_paragraph_index"])
    return obtained_memories, reasoning_trace, retrieval_counts


def run_full_roleplay_pipeline(
    character: str,
    user_query: str,
    stage1_prompt_fn: Callable,
    stage2_prompt_fn: Callable,
    llm_generator: Callable,
    retrieve_fn_map: Dict[str, Callable],
    generate_fn: Callable,
    on_trace_update: Callable = None,
    on_memory_update: Callable = None,
    max_steps: int = 3,
    max_retries: int = 5,
    retry_delay: float = 1.0,
) -> Tuple[str, str, List[Dict], Dict[str, int]]:
    from SR_EmotionRAG.memory_retrieval import DATABASE_PATH
    from SR_EmotionRAG.prompt_templates import build_roleplay_prompt
    from SR_EmotionRAG.utils import extract_json_from_response
    import json

    # Step 1–2: Memory collection
    final_memories, reasoning_trace, retrieval_counts = multistep_rag_loop_two_stage(
        character=character,
        user_query=user_query,
        stage1_prompt_fn=stage1_prompt_fn,
        stage2_prompt_fn=stage2_prompt_fn,
        llm_generator=llm_generator,
        retrieve_fn_map=retrieve_fn_map,
        on_trace_update=on_trace_update,
        on_memory_update=on_memory_update,
        max_steps=max_steps,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

    # Step 3: Prompt construction
    memory_fragments = [m["text"] for m in final_memories]
    with open(DATABASE_PATH / "system_prompts.json", "r", encoding="utf-8") as f:
        system_prompts = json.load(f)
    role_information = system_prompts[character]["system_prompt"]
    roleplay_prompt = build_roleplay_prompt(
        role=character,
        role_information=role_information,
        memory_fragments=memory_fragments,
        question=user_query
    )


    for attempt in range(max_retries + 1):
        try:
            raw_output = generate_fn(roleplay_prompt)
            break
        except Exception as e:
            print(f"[Stage 3 Retry {attempt + 1}] Generation failed: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                raw_output = ""

    # Try to extract a structured response (e.g., { "response": "..." })
    parsed = extract_json_from_response(raw_output)

    if parsed and "response" in parsed:
        character_response = parsed["response"]
    else:
        print("[Fallback] Using raw string output from model.")
        character_response = raw_output.strip()

    return roleplay_prompt, character_response, reasoning_trace, retrieval_counts

