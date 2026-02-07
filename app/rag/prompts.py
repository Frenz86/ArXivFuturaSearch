"""Prompt templates and context formatting for RAG."""

# Chain-of-Thought prompt template
COT_PROMPT_TEMPLATE = """You are an expert ML research assistant with deep knowledge of machine learning, deep learning, and AI research.

Your task is to answer questions based ONLY on the provided research paper excerpts. Follow this structured approach:

## Instructions:
1. First, identify which sources are most relevant to the question
2. Extract key information from each relevant source
3. Synthesize the information to form a coherent answer
4. Cite sources using [1], [2], etc. throughout your answer
5. If information is insufficient or contradictory, acknowledge this explicitly

## Few-Shot Examples:

**Example 1:**
Question: What are the main benefits of attention mechanisms?
Thinking: Sources [1] and [3] discuss attention mechanisms in detail. [1] mentions computational efficiency, while [3] focuses on interpretability.
Answer: According to the research, attention mechanisms provide two key benefits: First, they enable models to focus on relevant parts of the input, improving computational efficiency [1]. Second, they offer interpretability by revealing which parts of the input the model considers important for its predictions [3].

**Example 2:**
Question: How does batch normalization improve training?
Thinking: Source [2] provides evidence about batch normalization's effects on training stability and convergence.
Answer: Batch normalization improves training primarily by reducing internal covariate shift, which allows for higher learning rates and faster convergence [2]. However, the retrieved papers don't provide specific quantitative results about the magnitude of these improvements.

---

Now answer this question:

Question: {question}

Sources:
{context}

Let's think step by step:
1. Relevant sources:
2. Key information:
3. Synthesis:

Answer:"""


# Simple prompt template
SIMPLE_PROMPT_TEMPLATE = """You are an expert ML research assistant.

Rules:
- Answer ONLY using the provided sources.
- If the sources do not contain enough information, say: "I don't have enough evidence in the retrieved papers to answer that."
- Cite sources like [1], [2] in the answer.
- Be concise but insightful.

Question:
{question}

Sources:
{context}

Answer:"""


def format_context(retrieved: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context string.

    Args:
        retrieved: List of retrieval results with 'text' and 'meta'

    Returns:
        Formatted context string with citations
    """
    lines = []
    for i, r in enumerate(retrieved, start=1):
        title = r["meta"].get("title", "Untitled")
        link = r["meta"].get("link", "")
        lines.append(f"[{i}] {title}\nSOURCE: {link}\nEXCERPT: {r['text']}\n")
    return "\n".join(lines)


def build_prompt(question: str, retrieved: list[dict], use_cot: bool = True) -> str:
    """
    Build the RAG prompt with question and retrieved context.
    (Legacy function for backward compatibility)

    Args:
        question: User's question
        retrieved: List of retrieval results
        use_cot: Whether to use chain-of-thought prompting

    Returns:
        Complete prompt string
    """
    context = format_context(retrieved)

    template = COT_PROMPT_TEMPLATE if use_cot else SIMPLE_PROMPT_TEMPLATE
    return template.format(question=question, context=context)
