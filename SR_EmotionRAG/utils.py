import re
import json

def extract_json_from_response(response_content: str) -> dict | None:

    try:
        return json.loads(response_content.strip())
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        print("Raw content:\n", response_content)
        return None

def get_reasoning_output(prompt: str, llm_generator) -> dict | None:
    try:
        response = llm_generator(prompt)
        if not isinstance(response, list) or "generated_text" not in response[0]:
            raise ValueError("Unexpected response format from LLM generator.")
        
        generated = response[0]["generated_text"]
    except Exception as e:
        print(f"LLM generation failed: {e}")
        return None

    return extract_json_from_response(generated)

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


