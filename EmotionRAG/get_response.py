from EmotionRAG.memory_retrieval import retrieval
from EmotionRAG.prompt_templates import build_roleplay_prompt, SYSTEM_PROMPTS
from EmotionRAG.generation import openai_generator  # or huggingface_api_generator
from pathlib import Path

def get_response(
    character: str,
    question: str,
    database_path: str,
    method: str = "C-A",
    generate_fn=None
) -> str:
    """
    Generates a roleplay response for a character given a user question.

    Args:
        character (str): Character name.
        question (str): Interviewer's question.
        database_path (str): Path to memory DB.
        method (str): Retrieval method.
        generate_fn (callable): A function that takes a prompt and returns a string response.

    Returns:
        str: Character response from the LLM.
    """
    # === Step 1: Retrieve memory fragments ===
    memory_fragments, _ = retrieval(
        query_text=question,
        character=character,
        database_path=database_path,
        method=method
    )

    # === Step 2: Get system prompt ===
    if character not in SYSTEM_PROMPTS:
        raise ValueError(f"No system prompt found for character: {character}")
    role_information = SYSTEM_PROMPTS[character]["system_prompt"]

    # === Step 3: Build prompt ===
    prompt = build_roleplay_prompt(
        role=character,
        role_information=role_information,
        memory_fragments=memory_fragments,
        question=question
    )

    # === Step 4: Generate ===
    response = generate_fn(prompt)
    return response.strip()

