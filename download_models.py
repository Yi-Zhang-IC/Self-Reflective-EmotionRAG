from RAG.pipeline import run_full_roleplay_pipeline
from RAG.stage1_prompt import build_stage_1_prompt
from RAG.stage2_prompt import build_stage_2_prompt
from RAG.memory_retrieval import retrieve_top_k_memories, retrieve_top_k_hybrid_memories
from RAG.generation import openai_generator

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
        generation_backend="openhermes",        # can also be: "openai", "mistral", "deepseek"
        on_trace_update=on_trace_update,
        on_memory_update=on_memory_update
    )

    print("\n==================== Final Prompt ====================")
    print(roleplay_prompt)

    print("\n==================== Character Response ====================")
    print(character_response)

    print("\n==================== Retrieval Stats ====================")
    print(retrieval_counts)
