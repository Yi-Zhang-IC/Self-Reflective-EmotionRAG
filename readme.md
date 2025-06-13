# Self-Reflective Emotional RAG

This repository contains the full implementation of the **Self-Reflective Emotional RAG** framework, a memory-augmented architecture for simulating psychologically coherent characters via structured autobiographical memory, affective retrieval, and agentic reasoning.

---

## Repository Structure

### `ER_Emotion_RAG/`
Core implementation of the **Self-Reflective Emotional RAG** system. This includes:
- Multi-step agentic retrieval loop
- Emotional and semantic memory search
- Query decomposition and subquery planning
- Response generation pipeline

### `EmotionRAG/`
Replicates the baseline method from the [Emotional RAG](https://arxiv.org/abs/2402.13719) paper. Includes:
- Memory construction scripts based on GPT-generated summaries
- Affective and semantic retrieval modules as described in the original paper

### `scripts/`
Contains code for:
- Training the emotion embedding model
- Generating character responses from different RAG variants

### `prepare_datasets/`
Includes data preparation scripts and augmentation notebooks:
- Parse and preprocess the GoEmotions dataset
- GPT-based paraphrasing for emotion-balanced augmentation

### `data/`
Holds all emotion datasets:
- Augmented GoEmotions splits


### `database/`
Character-specific autobiographical memory banks and precomputed embeddings.

Each subdirectory (e.g., `harry_potter/`, `severus_snape/`, `draco_malfoy/`) contains the following:

- `memory.json`: A list of autobiographical memory entries describing emotionally and behaviorally relevant events (e.g., Draco Malfoy missing the Snitch due to arrogance).
- `embedding.pt`: Precomputed **semantic** embedding matrix used for meaning-based retrieval.
- `emotion_embeddings.pt`: Precomputed **emotion** embedding matrix trained on GoEmotions, used for affective matching.
- `gpt_emotion_embeddings.jsonl`: GPT-generated emotion-aligned embeddings from the baseline Emotion RAG system.


### `evaluation/`
Evaluation code and results:
- `character_response/`: Model-generated responses across MBTI and BFI interviews
- `embedding_tsne/`: Embedding visualization and retrieval evaluation notebooks

---

## Project Summary

The goal of this project is to build language agents that reflect coherent internal states by retrieving and reasoning over emotionally grounded autobiographical memory. This is achieved through a novel combination of:
- A 128-dimensional affective embedding space fine-tuned on GoEmotions
- Structured memory entries grounded in narrative biography
- Multi-step retrieval loops driven by semantic and emotional subqueries
- Evaluation via MBTI and BFI structured interviews using the InCharacter methodology

---

## Acknowledgements

Inspired by [Emotional RAG](https://arxiv.org/abs/2410.23041), [InCharacter](https://arxiv.org/abs/2310.17976), and related efforts in memory-based reasoning and emotion-aware agents.

---

For any questions or contributions, feel free to open an issue or contact the author.
