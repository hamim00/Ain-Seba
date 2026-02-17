"""
AinSeba - PDF Text Extractor
Extracts raw text from Bangladesh law PDFs using PyMuPDF (fitz).
Handles both well-structured and scanned PDFs gracefully.
"""

import fitz  # PyMuPDF
import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""
    page_number: int
    text: str
    char_count: int
    has_content: bool


@dataclass
class ExtractedDocument:
    """Represents the full extraction result from a PDF."""
    filename: str
    filepath: str
    total_pages: int
    pages: list[PageContent] = field(default_factory=list)
    full_text: str = ""
    extraction_warnings: list[str] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return len(self.full_text)

    @property
    def empty_pages(self) -> int:
        return sum(1 for p in self.pages if not p.has_content)


class PDFExtractor:
    """
    Extracts text from PDF files using PyMuPDF.
    
    Handles common issues with Bangladesh law PDFs:
    - Multi-column layouts
    - Headers/footers with page numbers
    - Mixed English/Bangla text
    - Scanned pages (flags them as warnings)
    """

    # Minimum characters on a page to consider it "has content"
    MIN_CONTENT_CHARS = 20

    def extract(self, pdf_path: str | Path) -> ExtractedDocument:
        """
        Extract all text from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ExtractedDocument with page-by-page and full text.

        Raises:
            FileNotFoundError: If PDF doesn't exist.
            RuntimeError: If PDF cannot be opened.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Extracting text from: {pdf_path.name}")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF '{pdf_path.name}': {e}")

        result = ExtractedDocument(
            filename=pdf_path.name,
            filepath=str(pdf_path),
            total_pages=len(doc),
        )

        all_text_parts = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text with layout preservation
            # "text" sort mode gives natural reading order
            text = page.get_text("text", sort=True)
            text = text.strip()

            char_count = len(text)
            has_content = char_count >= self.MIN_CONTENT_CHARS

            if not has_content and char_count > 0:
                result.extraction_warnings.append(
                    f"Page {page_num + 1}: Very little text ({char_count} chars) - "
                    f"may be scanned or image-only"
                )

            page_content = PageContent(
                page_number=page_num + 1,
                text=text,
                char_count=char_count,
                has_content=has_content,
            )
            result.pages.append(page_content)

            if has_content:
                all_text_parts.append(text)

        doc.close()

        result.full_text = "\n\n".join(all_text_parts)

        logger.info(
            f"Extracted {result.total_pages} pages, "
            f"{result.total_chars} chars, "
            f"{result.empty_pages} empty pages"
        )

        if result.extraction_warnings:
            for warning in result.extraction_warnings:
                logger.warning(warning)

        return result

    def extract_page_range(
        self, pdf_path: str | Path, start_page: int, end_page: int
    ) -> str:
        """
        Extract text from a specific page range (1-indexed, inclusive).

        Args:
            pdf_path: Path to the PDF file.
            start_page: First page (1-indexed).
            end_page: Last page (1-indexed, inclusive).

        Returns:
            Combined text from the specified page range.
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))

        start_idx = max(0, start_page - 1)
        end_idx = min(len(doc), end_page)

        texts = []
        for page_num in range(start_idx, end_idx):
            page = doc[page_num]
            text = page.get_text("text", sort=True).strip()
            if len(text) >= self.MIN_CONTENT_CHARS:
                texts.append(text)

        doc.close()
        return "\n\n".join(texts)

    def get_pdf_metadata(self, pdf_path: str | Path) -> dict:
        """
        Extract PDF metadata (title, author, creation date, etc.).

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dictionary of PDF metadata fields.
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        metadata = doc.metadata
        metadata["page_count"] = len(doc)
        doc.close()
        return metadata
