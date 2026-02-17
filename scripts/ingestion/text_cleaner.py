"""
AinSeba - Text Cleaner
Cleans and normalizes extracted text from Bangladesh law PDFs.
Handles common noise patterns found in legal documents.
"""

import re
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans raw extracted text from Bangladesh law PDFs.
    
    Handles:
    - Page headers/footers (e.g., "Page 42 of 156")
    - Gazette notification headers
    - Excessive whitespace and line breaks
    - Unicode normalization for Bangla text
    - Section numbering cleanup
    - Encoding artifacts
    """

    # Patterns to remove (compiled for performance)
    NOISE_PATTERNS = [
        # Page numbers in various formats
        r"(?m)^\s*[-—–]\s*\d+\s*[-—–]\s*$",
        r"(?m)^\s*Page\s+\d+\s*(of\s+\d+)?\s*$",
        r"(?m)^\s*\d+\s*$",  # Standalone numbers (page nums)

        # Common header/footer patterns in BD law docs
        r"(?m)^.*Bangladesh\s+Gazette.*$",
        r"(?m)^.*Extraordinary.*Gazette.*$",
        r"(?m)^.*Published\s+by\s+Authority.*$",
        r"(?m)^.*GOVERNMENT\s+OF.*BANGLADESH.*$",
        r"(?m)^.*Ministry\s+of\s+Law.*$",
        r"(?m)^.*Legislative\s+and\s+Parliamentary.*$",

        # Copyright / watermark patterns
        r"(?m)^.*bdlaws\.minlaw\.gov\.bd.*$",
        r"(?m)^.*www\.bdlaws\..*$",

        # Form feed and other control characters
        r"\f",
        r"\x0c",
    ]

    # Compiled patterns
    _compiled_patterns = None

    @classmethod
    def _get_compiled_patterns(cls) -> list[re.Pattern]:
        """Lazily compile regex patterns."""
        if cls._compiled_patterns is None:
            cls._compiled_patterns = [
                re.compile(p, re.IGNORECASE) for p in cls.NOISE_PATTERNS
            ]
        return cls._compiled_patterns

    def clean(self, text: str) -> str:
        """
        Apply all cleaning steps to extracted text.

        Args:
            text: Raw extracted text from PDF.

        Returns:
            Cleaned text ready for chunking.
        """
        if not text or not text.strip():
            return ""

        original_len = len(text)

        # Step 1: Remove noise patterns
        text = self._remove_noise_patterns(text)

        # Step 2: Fix encoding artifacts
        text = self._fix_encoding_artifacts(text)

        # Step 3: Normalize whitespace
        text = self._normalize_whitespace(text)

        # Step 4: Normalize section markers
        text = self._normalize_section_markers(text)

        # Step 5: Fix broken words (from PDF line breaks)
        text = self._fix_broken_words(text)

        # Step 6: Clean up Bangla text specifics
        text = self._clean_bangla_text(text)

        # Step 7: Final whitespace pass
        text = self._final_whitespace_cleanup(text)

        cleaned_len = len(text)
        reduction = ((original_len - cleaned_len) / original_len * 100) if original_len > 0 else 0

        logger.debug(
            f"Cleaned text: {original_len} → {cleaned_len} chars "
            f"({reduction:.1f}% reduction)"
        )

        return text

    def _remove_noise_patterns(self, text: str) -> str:
        """Remove header/footer/watermark noise."""
        for pattern in self._get_compiled_patterns():
            text = pattern.sub("", text)
        return text

    def _fix_encoding_artifacts(self, text: str) -> str:
        """Fix common encoding issues in PDF extraction."""
        replacements = {
            "\u2018": "'",   # Left single quote
            "\u2019": "'",   # Right single quote
            "\u201c": '"',   # Left double quote
            "\u201d": '"',   # Right double quote
            "\u2013": "-",   # En dash
            "\u2014": "—",   # Em dash (keep as em dash)
            "\u00a0": " ",   # Non-breaking space
            "\ufeff": "",    # BOM
            "\u200b": "",    # Zero-width space
            "\u200c": "",    # Zero-width non-joiner (careful with Bangla!)
            "\u200d": "",    # Zero-width joiner (careful with Bangla!)
        }

        # For Bangla text, we need to be careful with ZWNJ/ZWJ
        # They are used in Bangla conjuncts — only remove in English context
        # For now, keep them (remove from replacements if Bangla detected)

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize various whitespace characters."""
        # Replace tabs with spaces
        text = text.replace("\t", " ")

        # Collapse multiple spaces into one
        text = re.sub(r" {2,}", " ", text)

        # Remove trailing spaces on each line
        text = re.sub(r" +\n", "\n", text)

        # Collapse 3+ newlines into 2 (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def _normalize_section_markers(self, text: str) -> str:
        """
        Standardize section/chapter/part markers for consistent parsing.
        E.g., ensure "Section 42." format is consistent.
        """
        # Normalize "SECTION", "Section", "Sec." → "Section"
        text = re.sub(
            r"(?i)\b(?:SECTION|Sec\.?)\s*(\d+)",
            r"Section \1",
            text,
        )

        # Normalize chapter headers
        text = re.sub(
            r"(?i)\b(?:CHAPTER)\s+([IVXLCDM]+|\d+)",
            lambda m: f"CHAPTER {m.group(1).upper()}",
            text,
        )

        # Normalize part headers
        text = re.sub(
            r"(?i)\b(?:PART)\s+([IVXLCDM]+|\d+)",
            lambda m: f"PART {m.group(1).upper()}",
            text,
        )

        return text

    def _fix_broken_words(self, text: str) -> str:
        """
        Fix words broken across lines by PDF extraction.
        E.g., "employ-\nment" → "employment"
        """
        # Fix hyphenated line breaks
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Join lines that don't end with sentence-ending punctuation
        # (but keep paragraph breaks)
        text = re.sub(
            r"([a-zA-Z,;])\n([a-z])",
            r"\1 \2",
            text,
        )

        return text

    def _clean_bangla_text(self, text: str) -> str:
        """
        Handle Bangla-specific text issues.
        - Normalize Bangla numerals
        - Fix common Bangla encoding issues
        """
        # Map Bangla numerals to ASCII (for section numbers)
        bangla_digits = {
            "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
            "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
        }

        # Only convert Bangla digits that appear in section/chapter references
        for bangla, ascii_digit in bangla_digits.items():
            text = re.sub(
                rf"((?:Section|ধারা|Chapter|অধ্যায়|Part|পর্ব)\s*){bangla}",
                rf"\g<1>{ascii_digit}",
                text,
            )

        return text

    def _final_whitespace_cleanup(self, text: str) -> str:
        """Final pass to ensure clean text."""
        # Strip leading/trailing whitespace
        text = text.strip()

        # Ensure no more than 2 consecutive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text


def clean_text(text: str) -> str:
    """Convenience function for quick text cleaning."""
    cleaner = TextCleaner()
    return cleaner.clean(text)
