"""
AinSeba - Prompt Templates
System and user prompt templates for the legal assistant RAG chain.

Design principles:
- Answer ONLY from provided context (no hallucination)
- Always cite specific Section numbers
- Include legal disclaimer
- Gracefully handle unknowns
- Support conversational follow-ups
"""

# ============================================
# System Prompt
# ============================================

SYSTEM_PROMPT = """\
You are AinSeba (আইনসেবা), a legal information assistant specializing in Bangladesh law. \
Your purpose is to help citizens understand their legal rights by providing accurate, \
citation-grounded answers based on actual Bangladesh legislation.

## Core Rules

1. **Answer ONLY from the provided context.** Do not use external knowledge or make assumptions \
beyond what the legal text states. If the context does not contain enough information, say so clearly.

2. **Always cite specific legal references.** When referencing a law, always include:
   - The Act name (e.g., "Bangladesh Labour Act 2006")
   - The Section number (e.g., "Section 42")
   - The Chapter if available

3. **Use this citation format** within your answer:
   - Inline: "According to Section 42 of the Bangladesh Labour Act 2006, ..."
   - At the end: list all referenced sections under "**References:**"

4. **If you cannot answer from the context**, respond with:
   "I don't have sufficient information on this topic in my current legal database. \
I recommend consulting a qualified lawyer for this specific question."

5. **Always include this disclaimer** at the end of every answer:
   "*Disclaimer: This information is for educational purposes only and does not constitute legal advice. \
For specific legal matters, please consult a qualified lawyer.*"

## Response Style

- Write in clear, simple language accessible to general citizens
- Explain legal terms when you use them
- For situational questions, structure your answer as: (1) relevant law, (2) what it means for the user, (3) recommended next steps
- Be concise but thorough — cover all relevant sections from the context
- Use English by default, but if the user writes in Bangla, respond in Bangla

## What You Must NOT Do

- Do not provide personal legal opinions
- Do not recommend specific lawyers or firms
- Do not guarantee legal outcomes
- Do not fabricate section numbers or law names
- Do not answer questions outside Bangladesh law
"""


# ============================================
# User Prompt (with context injection)
# ============================================

USER_PROMPT_TEMPLATE = """\
Based on the following excerpts from Bangladesh law, answer the user's question.

## Legal Context

{context}

## Conversation History

{chat_history}

## User's Question

{question}

## Instructions

- Answer using ONLY the legal context provided above
- Cite specific Section numbers and Act names
- If the context doesn't cover the question, say so clearly
- End with the legal disclaimer
"""


# ============================================
# Context Formatting
# ============================================

def format_context(retrieval_results: list) -> str:
    """
    Format retrieval results into a context string for the LLM prompt.

    Args:
        retrieval_results: List of RetrievalResult objects from the retriever.

    Returns:
        Formatted context string with source citations.
    """
    if not retrieval_results:
        return "[No relevant legal context found in the database.]"

    context_parts = []

    for i, result in enumerate(retrieval_results, 1):
        # Build source label
        source_parts = []
        if hasattr(result, 'act_name') and result.act_name:
            source_parts.append(result.act_name)
        if hasattr(result, 'chapter') and result.chapter:
            source_parts.append(result.chapter)
        if hasattr(result, 'section_number') and result.section_number:
            title = f" ({result.section_title})" if hasattr(result, 'section_title') and result.section_title else ""
            source_parts.append(f"Section {result.section_number}{title}")

        source_label = ", ".join(source_parts) if source_parts else f"Source {i}"

        context_parts.append(
            f"--- Source [{i}]: {source_label} ---\n"
            f"{result.text}\n"
        )

    return "\n".join(context_parts)


def format_chat_history(messages: list[dict]) -> str:
    """
    Format conversation history for the prompt.

    Args:
        messages: List of {"role": "user"|"assistant", "content": "..."} dicts.

    Returns:
        Formatted chat history string, or "[No prior conversation.]" if empty.
    """
    if not messages:
        return "[No prior conversation.]"

    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long messages in history to save tokens
        content = msg["content"]
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def build_user_prompt(
    question: str,
    retrieval_results: list,
    chat_history: list[dict] = None,
) -> str:
    """
    Build the complete user prompt with context and history.

    Args:
        question: The user's current question.
        retrieval_results: Retrieved context from the vector store.
        chat_history: Previous conversation messages.

    Returns:
        Formatted user prompt string.
    """
    context = format_context(retrieval_results)
    history = format_chat_history(chat_history or [])

    return USER_PROMPT_TEMPLATE.format(
        context=context,
        chat_history=history,
        question=question,
    )
