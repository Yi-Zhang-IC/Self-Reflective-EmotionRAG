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

Step 1 — Sufficiency Check:
Carefully examine the user's question and the obtained memory information. Decide:
- Does the retrieved information fully and clearly answer the user's question?
- If yes, return an empty query list.
- If not, explain what is missing in your `reason`, and proceed to Step 2.

---

Step 2 — Generate Retrieval Queries (if needed):
If something is missing, generate up to {k} atomic queries to retrieve it.

Each query must be:
- Focused on one idea or emotion
- Self-contained (third person only, no "you")
- Assigned a correct `retrieval_type`

Retrieval types:
- `"semantic"`: for factual, causal, motivational, or introspective queries, including emotions tied to known people, events, or actions.
- `"hybrid"`: only if the query contains a **specific emotion word** (e.g., guilt, regret, anger, pride) and has **no clearly specified cause** — these use semantic retrieval followed by emotional filtering.

---

Output format:
Return only a JSON object in this structure:

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
