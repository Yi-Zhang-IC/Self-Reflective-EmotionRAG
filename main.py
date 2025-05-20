from RAG.pipeline import run_full_roleplay_pipeline
from RAG.stage1_prompt import build_stage_1_prompt
from RAG.stage2_prompt import build_stage_2_prompt
from RAG.memory_retrieval import retrieve_top_k_memories, retrieve_top_k_hybrid_memories
from RAG.generation import openai_generator
from pathlib import Path

MODEL = "openai"  # or "mistral", "deepseek"
ROOT_DIR = Path(__file__).resolve().parent

def lod_pipeline(generation_backend = MODEL):
       # Load model and generate
    if generation_backend == "openhermes":
        from RAG.generation import load_local_roleplay_model, local_llm_generator
        model = load_local_roleplay_model(str(ROOT_DIR / "models" / "OpenHermes-2.5-Mistral-7B"))
        generate_fn = lambda prompt: local_llm_generator(model, prompt)
    elif generation_backend == "deepseek":
        from RAG.generation import load_local_roleplay_model, local_llm_generator
        model = load_local_roleplay_model(str(ROOT_DIR / "models" / "DeepSeek-R1-Distill-Qwen-7B"))
        generate_fn = lambda prompt: local_llm_generator(model, prompt)
    elif generation_backend == "mistral":
        from RAG.generation import load_local_roleplay_model, local_llm_generator
        model = load_local_roleplay_model(str(ROOT_DIR / "models" / "Mistral-7B-v0.1"))
        generate_fn = lambda prompt: local_llm_generator(model, prompt)
    elif generation_backend == "openai":
        from RAG.generation import openai_generator
        generate_fn = lambda prompt: openai_generator(prompt)
    else:
        raise ValueError(f"Unknown backend: {generation_backend}")
    return generate_fn

# Load the model and generator function
generate_fn = lod_pipeline(generation_backend=MODEL)
retrievers = {
    "semantic": retrieve_top_k_memories,
    "hybrid": retrieve_top_k_hybrid_memories
}

# Streaming callback for reasoning steps
def on_trace_update(trace_step: dict):
    print(f"\n🧠 Reasoning Step: {trace_step['step']}")
    if "reason" in trace_step:
        print("  📌 Reason:", trace_step["reason"])
    if "queries" in trace_step:
        print("  🔍 Queries:")
        for q in trace_step["queries"]:
            print(f"    - {q['query']} ({q['retrieval_type']})")
    if "planned_queries" in trace_step:
        print("  🔧 Planned Queries:")
        for q in trace_step["planned_queries"]:
            print(f"    - {q['query']} ({q['retrieval_type']})")

# Streaming callback for each new memory
def on_memory_update(memory_item: dict):
    print(f"\n📚 Retrieved Memory (from query: {memory_item['source_query']}):")
    print(memory_item["text"])


if __name__ == "__main__":
    character = "severus_snape"
    user_query = "Why do you regret your actions toward Lily?"

    roleplay_prompt, character_response, reasoning_trace, retrieval_counts = run_full_roleplay_pipeline(
        character=character,
        user_query=user_query,
        stage1_prompt_fn=build_stage_1_prompt,
        stage2_prompt_fn=build_stage_2_prompt,
        llm_generator=openai_generator,
        retrieve_fn_map=retrievers,
        generate_fn=generate_fn,
        on_trace_update=on_trace_update,
        on_memory_update=on_memory_update
    )

    print("\n==================== Final Prompt ====================")
    print(roleplay_prompt)

    print("\n==================== Character Response ====================")
    print(character_response)

    print("\n==================== Retrieval Stats ====================")
    print(retrieval_counts)
