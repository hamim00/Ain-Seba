"""
AinSeba - Metadata-Aware Document Chunker
Splits Bangladesh law documents into semantically meaningful chunks
while preserving structural metadata (Act → Part → Chapter → Section).
"""

import re
import hashlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================
# Token Counter
# ============================================

try:
    import tiktoken
    _encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken (OpenAI tokenizer)."""
        return len(_encoding.encode(text))

except Exception:
    # tiktoken not installed OR can't download BPE data (network restricted)
    # Fall back to word-based approximation (~0.75 tokens per word for English)
    logger.warning(
        "tiktoken unavailable. Using approximate token counting. "
        "Install tiktoken and ensure network access for exact counts."
    )

    def count_tokens(text: str) -> int:
        """Approximate token count based on word splitting (~1.3 tokens/word)."""
        if not text:
            return 0
        words = text.split()
        return max(1, int(len(words) * 1.3))


# ============================================
# Data Models
# ============================================

@dataclass
class ChunkMetadata:
    """Metadata attached to each chunk for retrieval."""
    act_name: str
    act_id: str
    part: str = ""
    chapter: str = ""
    section_number: str = ""
    section_title: str = ""
    category: str = ""
    year: int = 0
    language: str = "english"
    page_numbers: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "act_name": self.act_name,
            "act_id": self.act_id,
            "part": self.part,
            "chapter": self.chapter,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "category": self.category,
            "year": self.year,
            "language": self.language,
            "page_numbers": self.page_numbers,
        }


@dataclass
class Chunk:
    """A single document chunk with text and metadata."""
    chunk_id: str
    text: str
    token_count: int
    metadata: ChunkMetadata
    chunk_index: int = 0  # Position in the document

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "token_count": self.token_count,
            "chunk_index": self.chunk_index,
            **self.metadata.to_dict(),
        }


# ============================================
# Structure Parser
# ============================================

class LegalStructureParser:
    """
    Parses the hierarchical structure of Bangladesh law documents.
    Detects: PART → CHAPTER → Section boundaries.
    """

    # Regex patterns for structure detection
    PART_PATTERN = re.compile(
        r"(?m)^(?:PART)\s+([IVXLCDM]+|\d+)\s*[:\-—.]?\s*(.*?)$",
        re.IGNORECASE,
    )
    CHAPTER_PATTERN = re.compile(
        r"(?m)^(?:CHAPTER)\s+([IVXLCDM]+|\d+)\s*[:\-—.]?\s*(.*?)$",
        re.IGNORECASE,
    )
    SECTION_PATTERN = re.compile(
        r"(?m)^Section\s+(\d+[A-Za-z]?)\s*[.\-—:]\s*(.*?)$",
        re.IGNORECASE,
    )
    # Alternate section pattern for numbered-only format: "42. Title here"
    NUMBERED_SECTION_PATTERN = re.compile(
        r"(?m)^(\d+)\.\s+([A-Z][^.]*?)(?:\.|$)",
    )

    def find_sections(self, text: str) -> list[dict]:
        """
        Find all section boundaries in the text.

        Returns:
            List of dicts with keys: type, number, title, start_pos, end_pos
        """
        markers = []

        # Find PARTs
        for match in self.PART_PATTERN.finditer(text):
            markers.append({
                "type": "part",
                "number": match.group(1).strip(),
                "title": match.group(2).strip(),
                "start_pos": match.start(),
            })

        # Find CHAPTERs
        for match in self.CHAPTER_PATTERN.finditer(text):
            markers.append({
                "type": "chapter",
                "number": match.group(1).strip(),
                "title": match.group(2).strip(),
                "start_pos": match.start(),
            })

        # Find Sections
        for match in self.SECTION_PATTERN.finditer(text):
            markers.append({
                "type": "section",
                "number": match.group(1).strip(),
                "title": match.group(2).strip(),
                "start_pos": match.start(),
            })

        # Sort by position in document
        markers.sort(key=lambda m: m["start_pos"])

        # Calculate end positions
        for i, marker in enumerate(markers):
            if i + 1 < len(markers):
                marker["end_pos"] = markers[i + 1]["start_pos"]
            else:
                marker["end_pos"] = len(text)

        return markers


# ============================================
# Main Chunker
# ============================================

class MetadataAwareChunker:
    """
    Chunks Bangladesh law documents while preserving legal structure metadata.
    
    Strategy:
    1. Parse document structure (Part → Chapter → Section)
    2. Try to keep each Section as one chunk if within token limit
    3. If a Section is too large, split it at paragraph boundaries
    4. Apply overlap between chunks for retrieval continuity
    5. Attach hierarchical metadata to each chunk
    """

    def __init__(
        self,
        chunk_size_tokens: int = 600,
        chunk_overlap_tokens: int = 100,
        min_chunk_tokens: int = 50,
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.parser = LegalStructureParser()

    def chunk_document(
        self,
        text: str,
        act_name: str,
        act_id: str,
        category: str = "",
        year: int = 0,
        language: str = "english",
    ) -> list[Chunk]:
        """
        Split a full law document into metadata-rich chunks.

        Args:
            text: Full cleaned text of the law document.
            act_name: Official name of the Act.
            act_id: Unique identifier for the Act.
            category: Legal category (e.g., "Employment", "Criminal Law").
            year: Year the law was enacted.
            language: Document language.

        Returns:
            List of Chunk objects with metadata.
        """
        logger.info(f"Chunking document: {act_name}")

        # Step 1: Parse document structure
        markers = self.parser.find_sections(text)

        if not markers:
            logger.warning(
                f"No structural markers found in '{act_name}'. "
                f"Falling back to paragraph-based chunking."
            )
            return self._fallback_chunk(
                text, act_name, act_id, category, year, language
            )

        logger.info(f"Found {len(markers)} structural markers")

        # Step 2: Build chunks respecting structure
        chunks = []
        current_part = ""
        current_chapter = ""

        for marker in markers:
            # Track hierarchy
            if marker["type"] == "part":
                current_part = f"Part {marker['number']}: {marker['title']}"
                continue
            elif marker["type"] == "chapter":
                current_chapter = f"Chapter {marker['number']}: {marker['title']}"
                continue

            # Process section content
            section_text = text[marker["start_pos"]:marker["end_pos"]].strip()
            section_tokens = count_tokens(section_text)

            base_metadata = ChunkMetadata(
                act_name=act_name,
                act_id=act_id,
                part=current_part,
                chapter=current_chapter,
                section_number=marker.get("number", ""),
                section_title=marker.get("title", ""),
                category=category,
                year=year,
                language=language,
            )

            if section_tokens <= self.chunk_size_tokens:
                # Section fits in one chunk
                if section_tokens >= self.min_chunk_tokens:
                    chunk = self._create_chunk(
                        text=section_text,
                        metadata=base_metadata,
                        chunk_index=len(chunks),
                    )
                    chunks.append(chunk)
            else:
                # Section too large — split at paragraph boundaries
                sub_chunks = self._split_large_section(
                    section_text, base_metadata, len(chunks)
                )
                chunks.extend(sub_chunks)

        # Step 3: Handle any text before the first marker (preamble)
        if markers and markers[0]["start_pos"] > 100:
            preamble = text[: markers[0]["start_pos"]].strip()
            if count_tokens(preamble) >= self.min_chunk_tokens:
                preamble_metadata = ChunkMetadata(
                    act_name=act_name,
                    act_id=act_id,
                    section_number="preamble",
                    section_title="Preamble / Preliminary",
                    category=category,
                    year=year,
                    language=language,
                )
                preamble_chunk = self._create_chunk(
                    text=preamble,
                    metadata=preamble_metadata,
                    chunk_index=0,
                )
                chunks.insert(0, preamble_chunk)
                # Re-index
                for i, chunk in enumerate(chunks):
                    chunk.chunk_index = i

        logger.info(
            f"Created {len(chunks)} chunks from '{act_name}' "
            f"(avg {sum(c.token_count for c in chunks) // max(len(chunks), 1)} tokens/chunk)"
        )

        return chunks

    def _split_large_section(
        self,
        text: str,
        base_metadata: ChunkMetadata,
        start_index: int,
    ) -> list[Chunk]:
        """
        Split a large section into smaller chunks at paragraph boundaries.
        Falls back to sentence-level splitting if paragraphs are too large.
        Applies overlap between chunks.
        """
        # First try paragraph-level splitting
        paragraphs = text.split("\n\n")

        # If there's only one "paragraph" that's too big, split by sentences
        if len(paragraphs) <= 1 and count_tokens(text) > self.chunk_size_tokens:
            paragraphs = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_text = ""
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = count_tokens(para)

            if current_tokens + para_tokens > self.chunk_size_tokens and current_text:
                # Save current chunk
                chunk = self._create_chunk(
                    text=current_text.strip(),
                    metadata=base_metadata,
                    chunk_index=start_index + len(chunks),
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_text)
                separator = "\n\n" if "\n\n" in text else " "
                current_text = overlap_text + separator + para if overlap_text else para
                current_tokens = count_tokens(current_text)
            else:
                separator = "\n\n" if "\n\n" in text else " "
                current_text += (separator + para if current_text else para)
                current_tokens += para_tokens

        # Don't forget the last chunk
        if current_text.strip() and count_tokens(current_text) >= self.min_chunk_tokens:
            chunk = self._create_chunk(
                text=current_text.strip(),
                metadata=base_metadata,
                chunk_index=start_index + len(chunks),
            )
            chunks.append(chunk)

        return chunks

    def _fallback_chunk(
        self,
        text: str,
        act_name: str,
        act_id: str,
        category: str,
        year: int,
        language: str,
    ) -> list[Chunk]:
        """
        Fallback chunking when no structural markers are found.
        Splits at paragraph boundaries with overlap.
        """
        logger.info("Using fallback paragraph-based chunking")

        base_metadata = ChunkMetadata(
            act_name=act_name,
            act_id=act_id,
            category=category,
            year=year,
            language=language,
        )

        return self._split_large_section(text, base_metadata, 0)

    def _get_overlap_text(self, text: str) -> str:
        """
        Extract the last N tokens of text for overlap.
        Tries to break at a sentence boundary.
        """
        if not text:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", text)

        overlap_text = ""
        overlap_tokens = 0

        for sentence in reversed(sentences):
            sent_tokens = count_tokens(sentence)
            if overlap_tokens + sent_tokens > self.chunk_overlap_tokens:
                break
            overlap_text = sentence + " " + overlap_text
            overlap_tokens += sent_tokens

        return overlap_text.strip()

    def _create_chunk(
        self, text: str, metadata: ChunkMetadata, chunk_index: int
    ) -> Chunk:
        """Create a Chunk object with a unique ID."""
        # Generate deterministic chunk ID from content + metadata
        hash_input = f"{metadata.act_id}:{chunk_index}:{text[:100]}"
        chunk_id = hashlib.md5(hash_input.encode()).hexdigest()[:12]

        return Chunk(
            chunk_id=f"{metadata.act_id}_{chunk_id}",
            text=text,
            token_count=count_tokens(text),
            metadata=metadata,
            chunk_index=chunk_index,
        )
