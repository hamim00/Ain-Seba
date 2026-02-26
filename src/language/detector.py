"""
AinSeba - Language Detection
Detects whether user queries are in Bangla, English, or mixed (Banglish).

Uses a multi-strategy approach:
1. Unicode script analysis (fastest, most reliable for Bangla)
2. langdetect as fallback
3. Banglish pattern detection (Latin-script Bangla words)
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    BANGLA = "bn"
    BANGLISH = "banglish"  # Bangla written in Latin script
    UNKNOWN = "unknown"


@dataclass
class DetectionResult:
    """Language detection result with confidence."""
    language: Language
    confidence: float          # 0.0 - 1.0
    bangla_ratio: float        # Ratio of Bangla characters
    has_bangla_script: bool    # Contains Bangla Unicode chars
    has_latin_script: bool     # Contains Latin chars
    is_mixed: bool             # Contains both scripts

    @property
    def is_bangla(self) -> bool:
        return self.language in (Language.BANGLA, Language.BANGLISH)

    @property
    def needs_translation(self) -> bool:
        """Whether the query needs English translation for retrieval."""
        return self.language in (Language.BANGLA, Language.BANGLISH)

    @property
    def response_language(self) -> str:
        """Suggested language for the response."""
        if self.language == Language.BANGLA:
            return "bn"
        elif self.language == Language.BANGLISH:
            return "bn"  # Respond in proper Bangla for Banglish input
        return "en"


# Bangla Unicode range: U+0980 - U+09FF
BANGLA_PATTERN = re.compile(r'[\u0980-\u09FF]')
LATIN_PATTERN = re.compile(r'[a-zA-Z]')

# Common Banglish words (Bangla words written in Latin script)
# These help detect Banglish when no Bangla Unicode is present
BANGLISH_MARKERS = {
    # Legal/formal terms
    "ain", "adalot", "bichar", "dondo", "shasti", "mamla",
    "apeal", "sajapraptho", "khotiporion", "ain-kanun",
    # Common words
    "ami", "amar", "amader", "apni", "apnar", "tumi", "tomar",
    "ki", "keno", "kothay", "kobe", "kivabe", "kon",
    "korte", "korbo", "korben", "kori", "kora",
    "hobe", "hoy", "hoyeche", "hoye",
    "ache", "achhe", "thake", "thaken",
    "jodi", "tahole", "kintu", "ebong", "ba", "othoba",
    "doya", "kore", "bolun", "bolen", "bolben",
    "shromik", "malik", "betan", "chuti",
    "poribesh", "niyom", "bidhi", "odhikar",
    # Question patterns
    "janaben", "janate", "jante", "chai",
    "parben", "paren", "parbo",
    "lagbe", "dorkar", "proyojon",
    # Nouns
    "sorkari", "bashai", "jaiga", "taka", "poisha",
    "kagoj", "potro", "din", "mash", "bochor",
    "manush", "lok", "meye", "chele",
    # Verbs
    "diben", "niben", "jaben", "asben",
    "dekhun", "shunun", "porun", "likhun",
}


def detect_language(text: str) -> DetectionResult:
    """
    Detect the language of a text query.

    Strategy:
    1. Count Bangla vs Latin characters
    2. If Bangla chars present -> Bangla or mixed
    3. If only Latin -> check for Banglish markers
    4. Fallback to langdetect library

    Args:
        text: User's query text.

    Returns:
        DetectionResult with language, confidence, and metadata.
    """
    if not text or not text.strip():
        return DetectionResult(
            language=Language.UNKNOWN,
            confidence=0.0,
            bangla_ratio=0.0,
            has_bangla_script=False,
            has_latin_script=False,
            is_mixed=False,
        )

    text_clean = text.strip()

    # Count character types
    bangla_chars = len(BANGLA_PATTERN.findall(text_clean))
    latin_chars = len(LATIN_PATTERN.findall(text_clean))
    total_alpha = bangla_chars + latin_chars

    has_bangla = bangla_chars > 0
    has_latin = latin_chars > 0
    is_mixed = has_bangla and has_latin

    bangla_ratio = bangla_chars / total_alpha if total_alpha > 0 else 0.0

    # Strategy 1: Bangla Unicode present
    if has_bangla:
        if bangla_ratio > 0.7:
            return DetectionResult(
                language=Language.BANGLA,
                confidence=min(0.95, 0.7 + bangla_ratio * 0.3),
                bangla_ratio=bangla_ratio,
                has_bangla_script=True,
                has_latin_script=has_latin,
                is_mixed=is_mixed,
            )
        elif bangla_ratio > 0.3:
            # Mixed Bangla and English
            return DetectionResult(
                language=Language.BANGLA,
                confidence=0.7,
                bangla_ratio=bangla_ratio,
                has_bangla_script=True,
                has_latin_script=has_latin,
                is_mixed=True,
            )
        else:
            # Mostly English with some Bangla words
            return DetectionResult(
                language=Language.ENGLISH,
                confidence=0.6,
                bangla_ratio=bangla_ratio,
                has_bangla_script=True,
                has_latin_script=True,
                is_mixed=True,
            )

    # Strategy 2: Only Latin script — check for Banglish
    if has_latin and not has_bangla:
        banglish_score = _detect_banglish(text_clean)
        if banglish_score > 0.3:
            return DetectionResult(
                language=Language.BANGLISH,
                confidence=min(0.85, 0.5 + banglish_score * 0.5),
                bangla_ratio=0.0,
                has_bangla_script=False,
                has_latin_script=True,
                is_mixed=False,
            )

    # Strategy 3: Fallback to langdetect
    try:
        from langdetect import detect as ld_detect
        detected = ld_detect(text_clean)
        if detected == "bn":
            return DetectionResult(
                language=Language.BANGLA,
                confidence=0.6,
                bangla_ratio=bangla_ratio,
                has_bangla_script=has_bangla,
                has_latin_script=has_latin,
                is_mixed=is_mixed,
            )
    except Exception:
        pass

    # Default to English
    return DetectionResult(
        language=Language.ENGLISH,
        confidence=0.8 if latin_chars > 0 else 0.5,
        bangla_ratio=0.0,
        has_bangla_script=False,
        has_latin_script=has_latin,
        is_mixed=False,
    )


def _detect_banglish(text: str) -> float:
    """
    Detect Banglish (Bangla written in Latin script).

    Scores based on presence of common Banglish words.

    Args:
        text: Latin-script text to check.

    Returns:
        Score from 0.0 (not Banglish) to 1.0 (definitely Banglish).
    """
    words = set(re.findall(r'[a-zA-Z]+', text.lower()))
    if not words:
        return 0.0

    matches = words.intersection(BANGLISH_MARKERS)
    ratio = len(matches) / len(words)

    # Boost score if multiple Banglish words found
    if len(matches) >= 3:
        ratio = min(1.0, ratio + 0.2)
    elif len(matches) >= 2:
        ratio = min(1.0, ratio + 0.1)

    return ratio
