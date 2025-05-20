import json
from pathlib import Path

# Load system prompts from file once (globally)
ROOT_DIR = Path(__file__).resolve().parents[1]
SYSTEM_PROMPTS_PATH = ROOT_DIR / "database" / "system_prompts.json"
with open(Path(SYSTEM_PROMPTS_PATH), "r", encoding="utf-8") as f:
    # Load the system prompts from the JSON file
    SYSTEM_PROMPTS = json.load(f)

def build_stage_1_prompt(character: str, user_query: str, k: int = 3) -> str:
    character_info = SYSTEM_PROMPTS[character]["system_prompt"]

    return f"""You are a retrieval query planner for the character "{character}". Your job is to create up to {k} focused memory search queries to help retrieve relevant information from the character's past.

Character background:
{character_info.strip()}

---

Your task:

You are not answering the user's question directly. You are planning how to **search a memory database** to help answer it. Follow these steps:

---

Step 1 — Query Decomposition:
Break down the user query into up to {k} **atomic, retrieval-focused subqueries**.

- If the original query is already useful, you may keep it (rephrased), you can add complementary subquery to cover related angles.
- If the query is vague or abstract (e.g., "How did Snape view his life?"), interpret it as a prompt to explore emotional or motivational themes.
- Queries must be self-contained and atomic — targeting exactly one emotion, cause, or idea, using simple phrasing.

---

Step 2 — Assign Retrieval Type:
Assign a `retrieval_type` to each subquery:

- `"semantic"`: use for factual, causal, motivational, or introspective questions, or when emotion is clearly tied to a known person, event, or action.
- `"hybrid"`: use only when the query contains a **strong emotion word** (e.g., guilt, regret, anger) but has **no clearly defined cause**.

---

Step 3 — Output:
Return only a JSON array of objects with a `query` and `retrieval_type`.

Example:

User query: "What made you feel regret and sadness?"

Output:

[
  {{
    "query": "What made Snape feel regret?",
    "retrieval_type": "hybrid"
  }},
  {{
    "query": "Did Snape ever feel pride?",
    "retrieval_type": "hybrid"
  }}
]

---

User query:
{user_query}
"""
