import json
from pathlib import Path

# Load system prompts from file once (globally)
ROOT_DIR = Path(__file__).resolve().parents[1]
SYSTEM_PROMPTS_PATH = ROOT_DIR / "database" / "system_prompts.json"
with open(Path(SYSTEM_PROMPTS_PATH), "r", encoding="utf-8") as f:
    # Load the system prompts from the JSON file
    SYSTEM_PROMPTS = json.load(f)

def build_roleplay_prompt(role: str, role_information: str, memory_fragments: list[str], question: str) -> str:
    memory_text = "\n".join(f"- {frag.strip()}" for frag in memory_fragments)

    return f"""
You are {role}. Please answer the interviewer's question using the tone, personality, and knowledge of {role}. Stay in character.
[Role Information]
{role_information.strip()}

---
Here is the interviewer's question:
Interviewer: {question.strip()}

These are the memories you recalled in response to the question:
---
{memory_text}
---

Please answer as {role} in a few sentences. Keep your tone authentic and consistent with the character. You are not allowed to include your name and in the response.
{role}:
"""
