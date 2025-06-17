import streamlit as st
from pathlib import Path
import sys
import json


# Add SR_EmotionRAG to import path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from SR_EmotionRAG.pipeline import run_full_roleplay_pipeline
from SR_EmotionRAG.memory_retrieval import (
    retrieve_top_k_memories,
    retrieve_top_k_emotional_memories,
    retrieve_top_k_hybrid_memories
)
from SR_EmotionRAG.generation import load_pipeline, openai_generator
from SR_EmotionRAG.stage1_prompt import build_stage_1_prompt
from SR_EmotionRAG.stage2_prompt import build_stage_2_prompt
from SR_EmotionRAG.utils import on_memory_update, on_trace_update

# Load question metadata
question_data = json.load(open(ROOT_DIR / "evaluation" / "16Personalities.json", encoding="utf-8"))
default_question = question_data["questions"]["18"]["rewritten_en"]

# Character and backend choices
characters = [
    "albus_dumbledore", "draco_malfoy", "harry_potter", "hermione_granger",
    "luna_lovegood", "minerva_mcgonagall", "ron_weasley", "severus_snape"
]
backends = ["llama3", "qwen", "deepseek"]
retrievers = {
    "semantic": retrieve_top_k_memories,
    "emotional": retrieve_top_k_emotional_memories,
    "hybrid": retrieve_top_k_hybrid_memories
}

st.set_page_config(page_title="Self-Reflective Emotional RAG Demo")
st.title("🧠 Self-Reflective Emotional RAG")

# --- Sidebar config
character = st.selectbox("Choose a character", characters)
backend = st.selectbox("Choose a generation backend", backends)
user_question = st.text_area("Ask your question", value=default_question)

if st.button("Generate Response"):
    st.info("Generating response...")

    try:
        generate_fn = load_pipeline(backend)

        prompt, response, trace, retrieval_counts, memory_by_step = run_full_roleplay_pipeline(
            max_steps=2,
            character=character,
            user_query=user_question,
            stage1_prompt_fn=build_stage_1_prompt,
            stage2_prompt_fn=build_stage_2_prompt,
            llm_generator=openai_generator,
            retrieve_fn_map=retrievers,
            generate_fn=generate_fn,
            on_trace_update=on_trace_update,
            on_memory_update=on_memory_update
        )

        st.subheader("💬 Character Response")
        st.write(response.strip())

        with st.expander("📚 Retrieval Counts"):
            st.json(retrieval_counts)

        with st.expander("📖 Roleplay Prompt (Final input to LLM)"):
            st.text_area(label="Prompt", value=prompt, height=250)


        st.subheader("🧠 Self-Reflective Reasoning Trace")
        for step_info in trace:
            step = step_info.get("step", "unknown").replace("_", " ").title()
            with st.expander(f"🔹 {step}"):
                if "original_query" in step_info:
                    st.markdown(f"**User Question:** {step_info['original_query']}")
                if "reason" in step_info:
                    st.markdown(f"**Agent Reflection:** {step_info['reason']}")
                subqs = step_info.get("planned_queries", step_info.get("queries", []))
                if subqs:
                    subq_table = {
                        "Subquery": [q["query"] for q in subqs],
                        "Retrieval Type": [q["retrieval_type"] for q in subqs],
                    }
                    st.table(subq_table)

        st.subheader("🧩 Retrieved Memory Fragments by Step")
        for step_key, mems in memory_by_step.items():
            with st.expander(f"📌 Memories Retrieved During: {step_key.replace('_', ' ').title()}"):
                for i, mem in enumerate(mems, 1):
                    st.markdown(f"**{i}. From query:** _{mem['source_query']}_")
                    st.markdown(mem["text"])
                    st.markdown("---")

    except Exception as e:
        st.error(f"Generation failed: {e}")
