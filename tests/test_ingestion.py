"""
AinSeba - Unit Tests for Phase 1 (Ingestion Pipeline)

Run with: pytest tests/test_ingestion.py -v
"""

import sys
import json
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingestion.text_cleaner import TextCleaner
from scripts.ingestion.chunker import (
    MetadataAwareChunker,
    LegalStructureParser,
    count_tokens,
    ChunkMetadata,
)
from scripts.ingestion.quality_report import QualityReporter


# ============================================
# Test: Text Cleaner
# ============================================

class TestTextCleaner:
    """Tests for the TextCleaner class."""

    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_removes_page_numbers(self):
        text = "Some content here\n- 42 -\nMore content"
        cleaned = self.cleaner.clean(text)
        assert "- 42 -" not in cleaned
        assert "Some content here" in cleaned

    def test_removes_standalone_numbers(self):
        text = "Legal text here.\n\n42\n\nMore legal text."
        cleaned = self.cleaner.clean(text)
        assert "\n42\n" not in cleaned

    def test_removes_gazette_headers(self):
        text = "Bangladesh Gazette, Extra, September 15, 2023\nActual content here"
        cleaned = self.cleaner.clean(text)
        assert "Gazette" not in cleaned
        assert "Actual content here" in cleaned

    def test_collapses_excessive_whitespace(self):
        text = "Line one.\n\n\n\n\nLine two."
        cleaned = self.cleaner.clean(text)
        assert "\n\n\n" not in cleaned
        assert "Line one." in cleaned and "Line two." in cleaned

    def test_fixes_broken_words(self):
        text = "employ-\nment is important"
        cleaned = self.cleaner.clean(text)
        assert "employment" in cleaned

    def test_normalizes_section_markers(self):
        text = "SECTION 42. Title of section"
        cleaned = self.cleaner.clean(text)
        assert "Section 42" in cleaned

    def test_preserves_meaningful_content(self):
        text = "Section 10. Every worker shall be entitled to annual leave."
        cleaned = self.cleaner.clean(text)
        assert "worker" in cleaned
        assert "annual leave" in cleaned

    def test_handles_empty_input(self):
        assert self.cleaner.clean("") == ""
        assert self.cleaner.clean("   ") == ""

    def test_removes_bdlaws_watermark(self):
        text = "Content here\nbdlaws.minlaw.gov.bd\nMore content"
        cleaned = self.cleaner.clean(text)
        assert "bdlaws" not in cleaned


# ============================================
# Test: Legal Structure Parser
# ============================================

class TestLegalStructureParser:
    """Tests for the LegalStructureParser class."""

    def setup_method(self):
        self.parser = LegalStructureParser()

    def test_finds_parts(self):
        text = "PART I\nPRELIMINARY\n\nSome content here."
        markers = self.parser.find_sections(text)
        parts = [m for m in markers if m["type"] == "part"]
        assert len(parts) >= 1
        assert parts[0]["number"] == "I"

    def test_finds_chapters(self):
        text = "CHAPTER III\nWAGES AND BENEFITS\n\nContent."
        markers = self.parser.find_sections(text)
        chapters = [m for m in markers if m["type"] == "chapter"]
        assert len(chapters) >= 1
        assert chapters[0]["number"] == "III"

    def test_finds_sections(self):
        text = "Section 42. Penalty for violations\nContent of the section."
        markers = self.parser.find_sections(text)
        sections = [m for m in markers if m["type"] == "section"]
        assert len(sections) >= 1
        assert sections[0]["number"] == "42"

    def test_finds_multiple_sections(self):
        text = (
            "Section 1. Short title\nThis act may be called...\n\n"
            "Section 2. Definitions\nIn this act...\n\n"
            "Section 3. Application\nThis act applies to...\n"
        )
        markers = self.parser.find_sections(text)
        sections = [m for m in markers if m["type"] == "section"]
        assert len(sections) == 3

    def test_markers_sorted_by_position(self):
        text = (
            "PART I\nPRELIMINARY\n\n"
            "CHAPTER I\nINTRODUCTION\n\n"
            "Section 1. Title\nContent.\n\n"
            "Section 2. Definitions\nContent.\n"
        )
        markers = self.parser.find_sections(text)
        positions = [m["start_pos"] for m in markers]
        assert positions == sorted(positions)

    def test_handles_no_structure(self):
        text = "Just a plain paragraph with no legal structure markers."
        markers = self.parser.find_sections(text)
        assert len(markers) == 0


# ============================================
# Test: Metadata-Aware Chunker
# ============================================

class TestMetadataAwareChunker:
    """Tests for the MetadataAwareChunker class."""

    def setup_method(self):
        self.chunker = MetadataAwareChunker(
            chunk_size_tokens=200,
            chunk_overlap_tokens=50,
            min_chunk_tokens=10,
        )

    def test_chunks_simple_document(self):
        text = (
            "Section 1. Short title\n"
            "This Act may be called the Test Act, 2024.\n\n"
            "Section 2. Definitions\n"
            "In this Act, 'worker' means any person employed.\n\n"
            "Section 3. Application\n"
            "This Act applies to all establishments.\n"
        )
        chunks = self.chunker.chunk_document(
            text=text,
            act_name="Test Act 2024",
            act_id="test_act_2024",
        )
        assert len(chunks) >= 1
        assert all(c.metadata.act_name == "Test Act 2024" for c in chunks)
        assert all(c.metadata.act_id == "test_act_2024" for c in chunks)

    def test_chunk_has_metadata(self):
        text = (
            "CHAPTER I\nINTRODUCTION\n\n"
            "Section 1. Short title\n"
            "This Act may be called the Test Act.\n"
        )
        chunks = self.chunker.chunk_document(
            text=text,
            act_name="Test Act",
            act_id="test_act",
            category="Employment",
            year=2024,
        )
        if chunks:
            chunk = chunks[0]
            assert chunk.metadata.act_name == "Test Act"
            assert chunk.metadata.category == "Employment"
            assert chunk.metadata.year == 2024

    def test_chunk_ids_are_unique(self):
        text = (
            "Section 1. First section\n" + "Content. " * 50 + "\n\n"
            "Section 2. Second section\n" + "Content. " * 50 + "\n"
        )
        chunks = self.chunker.chunk_document(
            text=text, act_name="Test", act_id="test"
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_respects_max_chunk_size(self):
        # Create a long section that must be split
        long_text = "Section 1. Long section\n" + ("This is a test sentence. " * 200)
        chunks = self.chunker.chunk_document(
            text=long_text, act_name="Test", act_id="test"
        )
        # Most chunks should be within 1.5x the target size
        for chunk in chunks:
            assert chunk.token_count < self.chunker.chunk_size_tokens * 2

    def test_fallback_chunking_no_markers(self):
        text = "Just a plain paragraph. " * 100
        chunks = self.chunker.chunk_document(
            text=text, act_name="Test", act_id="test"
        )
        assert len(chunks) >= 1

    def test_skips_tiny_chunks(self):
        text = (
            "Section 1. A\nB\n\n"  # Very short
            "Section 2. Substantial content\n" + "Legal text here. " * 30 + "\n"
        )
        chunks = self.chunker.chunk_document(
            text=text, act_name="Test", act_id="test"
        )
        for chunk in chunks:
            assert chunk.token_count >= self.chunker.min_chunk_tokens


# ============================================
# Test: Token Counter
# ============================================

class TestTokenCounter:
    """Tests for the token counting function."""

    def test_counts_tokens(self):
        text = "This is a simple sentence."
        tokens = count_tokens(text)
        assert tokens > 0
        assert tokens < 20

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_longer_text_more_tokens(self):
        short = "Hello"
        long = "Hello world, this is a longer sentence with more tokens."
        assert count_tokens(long) > count_tokens(short)


# ============================================
# Test: Quality Reporter
# ============================================

class TestQualityReporter:
    """Tests for the QualityReporter class."""

    def setup_method(self):
        self.reporter = QualityReporter()

    def _make_chunks(self, n: int = 10, token_count: int = 400) -> list[dict]:
        """Create dummy chunks for testing."""
        return [
            {
                "chunk_id": f"test_{i}",
                "text": "Sample text " * (token_count // 3),
                "token_count": token_count,
                "chunk_index": i,
                "act_name": "Test Act",
                "act_id": "test_act",
                "chapter": f"Chapter {i // 3 + 1}",
                "section_number": str(i + 1),
                "section_title": f"Section {i + 1} Title",
                "category": "Test",
            }
            for i in range(n)
        ]

    def test_generates_report(self):
        chunks = self._make_chunks()
        report = self.reporter.generate_report(chunks, "Test Act")
        assert "summary" in report
        assert "token_distribution" in report
        assert "metadata_completeness" in report
        assert "quality_score" in report

    def test_summary_stats(self):
        chunks = self._make_chunks(n=5, token_count=300)
        report = self.reporter.generate_report(chunks)
        assert report["summary"]["total_chunks"] == 5
        assert report["summary"]["avg_tokens_per_chunk"] == 300

    def test_detects_short_chunks(self):
        chunks = self._make_chunks(n=5, token_count=400)
        chunks.append({
            "chunk_id": "short_one",
            "text": "Too short",
            "token_count": 10,
            "chunk_index": 5,
            "act_name": "Test",
            "act_id": "test",
        })
        report = self.reporter.generate_report(chunks)
        short_issues = [i for i in report["content_issues"] if i["type"] == "very_short"]
        assert len(short_issues) >= 1

    def test_quality_score_range(self):
        chunks = self._make_chunks()
        report = self.reporter.generate_report(chunks)
        score = report["quality_score"]["score"]
        assert 0 <= score <= 100

    def test_handles_empty_input(self):
        report = self.reporter.generate_report([])
        assert "error" in report

    def test_saves_to_file(self, tmp_path):
        chunks = self._make_chunks()
        output = tmp_path / "test_report.json"
        self.reporter.generate_report(chunks, output_path=output)
        assert output.exists()
        with open(output) as f:
            saved = json.load(f)
        assert "summary" in saved


# ============================================
# Test: Integration (End-to-End)
# ============================================

class TestIntegration:
    """Integration tests that verify the pipeline end-to-end."""

    def test_clean_then_chunk(self):
        """Verify cleaning and chunking work together."""
        cleaner = TextCleaner()
        chunker = MetadataAwareChunker(chunk_size_tokens=200, min_chunk_tokens=10)

        raw_text = (
            "Bangladesh Gazette, Extra\n"  # Should be cleaned
            "- 1 -\n"  # Page number, should be cleaned
            "Section 1. Short title\n"
            "This Act may be called the Test Act, 2024.\n\n"
            "Section 2. Definitions\n"
            "In this Act:\n"
            "(a) 'employer' means any person who employs workers;\n"
            "(b) 'worker' means any person employed in an establishment;\n"
            "(c) 'establishment' means any shop or factory.\n\n"
            "Section 3. Application\n"
            "This Act applies to all establishments in Bangladesh.\n"
        )

        cleaned = cleaner.clean(raw_text)
        assert "Gazette" not in cleaned

        chunks = chunker.chunk_document(
            text=cleaned,
            act_name="Test Act",
            act_id="test",
        )
        assert len(chunks) >= 1

        # Verify metadata is present
        for chunk in chunks:
            assert chunk.metadata.act_name == "Test Act"
            assert chunk.text.strip()  # No empty chunks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
