"""
AinSeba - Query Translation
Translates Bangla/Banglish queries to English for retrieval,
and translates English responses back to Bangla for the user.

Uses GPT-4o-mini for accurate legal term translation.
"""

import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


# Translation prompts
QUERY_TRANSLATION_PROMPT = """\
You are a translator for a Bangladesh legal aid system. \
Translate the following query from {source_lang} to English. \
Keep legal terms accurate. Produce ONLY the translated text, nothing else.

Query: {query}
"""

RESPONSE_TRANSLATION_PROMPT = """\
You are a translator for a Bangladesh legal aid system. \
Translate the following legal response from English to Bangla (Bengali). \

Rules:
- Use standard Bangla (not colloquial/regional dialect)
- Keep legal section references in English (e.g., "Section 42" stays as "Section 42")
- Keep Act names in both English and Bangla on first mention, e.g., "বাংলাদেশ শ্রম আইন ২০০৬ (Bangladesh Labour Act 2006)"
- Keep the disclaimer at the end
- Preserve all formatting (bold, bullet points, etc.)

Response to translate:
{response}
"""

BANGLISH_TO_ENGLISH_PROMPT = """\
You are a translator for a Bangladesh legal aid system. \
The following query is in Banglish (Bangla written in Latin/English script). \
Translate it to proper English while preserving the legal intent. \
Produce ONLY the translated text, nothing else.

Banglish query: {query}
"""


class QueryTranslator:
    """
    Translates queries and responses between Bangla and English.
    
    Pipeline for Bangla queries:
    1. User query (Bangla/Banglish) -> translate to English
    2. Retrieve using English query (embeddings work better in English)
    3. Generate answer in English
    4. Translate answer to Bangla
    
    Uses GPT-4o-mini for translation — fast, cheap, good quality.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
    ):
        """
        Args:
            api_key: OpenAI API key.
            model: Model to use for translation.
        """
        if not api_key:
            raise ValueError("OpenAI API key is required for translation.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.total_translation_tokens = 0

    def translate_query_to_english(
        self,
        query: str,
        source_language: str = "bn",
    ) -> str:
        """
        Translate a Bangla or Banglish query to English for retrieval.

        Args:
            query: The user's query in Bangla/Banglish.
            source_language: Source language code ("bn" or "banglish").

        Returns:
            English translation of the query.
        """
        if source_language == "banglish":
            prompt = BANGLISH_TO_ENGLISH_PROMPT.format(query=query)
        else:
            prompt = QUERY_TRANSLATION_PROMPT.format(
                source_lang="Bangla (Bengali)",
                query=query,
            )

        translation = self._call_llm(prompt)
        logger.info(f"Query translated: '{query[:50]}...' -> '{translation[:50]}...'")
        return translation

    def translate_response_to_bangla(self, response: str) -> str:
        """
        Translate an English legal response to Bangla.

        Args:
            response: English response text from the RAG chain.

        Returns:
            Bangla translation of the response.
        """
        prompt = RESPONSE_TRANSLATION_PROMPT.format(response=response)
        translation = self._call_llm(prompt, max_tokens=2000)
        logger.info(f"Response translated to Bangla ({len(translation)} chars)")
        return translation

    def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 500,
    ) -> str:
        """
        Make a translation API call.

        Args:
            prompt: The translation prompt.
            max_tokens: Maximum response tokens.

        Returns:
            Translated text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistent translation
                max_tokens=max_tokens,
            )

            self.total_translation_tokens += response.usage.total_tokens
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Return original text on failure
            return prompt.split("Query: ")[-1].split("Response to translate:")[-1].strip()
