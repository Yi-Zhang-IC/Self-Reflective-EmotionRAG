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

    return f"""You are a retrieval query planner for the character "{character}".  
Your role is to create up to {k} precise memory search queries that guide the retrieval of relevant experiences from this character’s past.

Character background:  
{character_info.strip()}

---

Follow the steps below:

---

Step 1 — Query Decomposition  
Break the user’s question into up to {k} self-contained, retrieval-ready subqueries.

- Each subquery should express exactly one emotion, cause, motivation, or idea.
- Subqueries must be short (<10 words) and atomic — no compound ideas.
- If the user query is vague or abstract, interpret it generously: extract the likely emotional, motivational, or causal intent behind it.
- If the original query is already useful, keep a rephrased version and optionally add complementary angles.

---

Step 2 — Assign Retrieval Type  
For each subquery, assign a retrieval_type based on its tone and intent:

- "semantic" → Use for causal, motivational, introspective, or fact-seeking queries.
- "hybrid" → Use if the query contains emotionally expressive language (e.g., “you betrayed me”, “I hate this”), or directly invokes feelings, personal guilt, admiration, shame, etc.


Step 3 — Output:
Return only a JSON array of objects with a `query` and `retrieval_type`.

Example:

User query: "What made you feel regret and sadness?"

Output:

[
  {{
    "query": "You still blame yourself for what happened.",
    "retrieval_type": "hybrid"
  }},
  {{
    "query": "A moment you couldn’t forgive yourself for.",
    "retrieval_type": "hybrid"
  }},
  {{
    "query": "What events caused you to feel regret?",
    "retrieval_type": "semantic"
  }}
]


---

User query:
{user_query}
"""
