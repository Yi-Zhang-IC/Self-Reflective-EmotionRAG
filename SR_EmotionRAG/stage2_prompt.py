import json
from pathlib import Path

# Load system prompts from file once (globally)
ROOT_DIR = Path(__file__).resolve().parents[1]
SYSTEM_PROMPTS_PATH = ROOT_DIR / "database" / "system_prompts.json"
with open(Path(SYSTEM_PROMPTS_PATH), "r", encoding="utf-8") as f:
    # Load the system prompts from the JSON file
    SYSTEM_PROMPTS = json.load(f)

def build_stage_2_prompt(character: str, query: str, obtained_memories: list, k: int = 3) -> str:
    if character not in SYSTEM_PROMPTS:
        raise ValueError(f"Character '{character}' not found in system_prompts.json.")

    character_info = SYSTEM_PROMPTS[character]["system_prompt"]

    prompt_header = f"""You are a memory-reasoning assistant helping to plan memory retrieval for the character "{character}". Your goal is to determine whether the current information is sufficient to answer the user's question. If not, generate new search queries to fill in missing knowledge.

Character background:
{character_info.strip()}

---

Your task consists of two parts:

---

---

Step 1 — Sufficiency Check:
Carefully examine the user's question and the obtained memory information. Decide:
- Does the retrieved information fully and clearly answer the user's question?
- If yes, return an empty query list.
- If not, explain what is missing in your `reason`, and proceed to Step 2.

---

Step 2 — Generate Retrieval Queries (if needed):
If something is missing, generate up to {k} atomic queries to retrieve it.

Each query must be:
- Each subquery should express exactly one emotion, cause, motivation, or idea.
- Subqueries must be short (<10 words) and atomic — no compound ideas.
- Assigned a correct `retrieval_type`

---

Step 3 — Assign Retrieval Type  
For each subquery, assign a retrieval_type based on its tone and intent:

- "semantic" → Use for causal, motivational, introspective, or fact-seeking queries.
- "hybrid" → Use if the query contains emotionally expressive language (e.g., “you betrayed me”, “I hate this”), or directly invokes feelings, personal guilt, admiration, shame, etc.

---

Step 4 — Output:
Return only a JSON array of objects with with the following structure:

{{
  "reason": "<your explanation of what is missing or why retrieval is needed>",
  "queries": [
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
}}

assert isinstance(output, dict)
assert "reason" in output and "queries" in output

---

User query:
{query}

Obtained information:
"""

    if not obtained_memories:
        obtained_info = "(none)\n"
    else:
        obtained_info = ""
        for mem in obtained_memories:
            obtained_info += f'[From query: "{mem["source_query"]}"]\n{mem["text"].strip()}\n\n'

    return prompt_header + obtained_info
