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


