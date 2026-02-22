"""
AinSeba - Ingestion Pipeline Orchestrator
Coordinates the full flow: PDF → Extract → Clean → Chunk → Save → Report.
"""

import json
import csv
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingestion.pdf_extractor import PDFExtractor
from scripts.ingestion.text_cleaner import TextCleaner
from scripts.ingestion.chunker import MetadataAwareChunker, count_tokens
from scripts.ingestion.quality_report import QualityReporter, print_report_summary
from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    LAW_REGISTRY,
    get_law_by_id,
)

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Full document ingestion pipeline for Bangladesh law documents.
    
    Pipeline stages:
    1. PDF Text Extraction (PyMuPDF)
    2. Text Cleaning (noise removal, normalization)
    3. Metadata-Aware Chunking (Section-level)
    4. Export to JSON & CSV
    5. Quality Report Generation
    """

    def __init__(
        self,
        raw_dir: Path = RAW_DATA_DIR,
        output_dir: Path = PROCESSED_DATA_DIR,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.extractor = PDFExtractor()
        self.cleaner = TextCleaner()
        self.chunker = MetadataAwareChunker(
            chunk_size_tokens=chunk_size,
            chunk_overlap_tokens=chunk_overlap,
        )
        self.reporter = QualityReporter()

    def process_single_pdf(
        self,
        pdf_path: str | Path,
        law_config: dict | None = None,
    ) -> dict:
        """
        Process a single PDF through the full pipeline.

        Args:
            pdf_path: Path to the PDF file.
            law_config: Optional law registry entry with metadata.
                       If None, metadata is inferred from filename.

        Returns:
            Dictionary with processing results and statistics.
        """
        pdf_path = Path(pdf_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {pdf_path.name}")
        logger.info(f"{'='*60}")

        # ---- Stage 1: Extract ----
        logger.info("[1/5] Extracting text from PDF...")
        extracted = self.extractor.extract(pdf_path)
        logger.info(
            f"  Extracted {extracted.total_pages} pages, "
            f"{extracted.total_chars:,} characters"
        )

        if extracted.total_chars < 100:
            logger.error(f"  ⚠ Very little text extracted. PDF may be scanned/image-only.")
            return {
                "status": "failed",
                "reason": "insufficient_text",
                "filename": pdf_path.name,
                "chars_extracted": extracted.total_chars,
            }

        # ---- Stage 2: Clean ----
        logger.info("[2/5] Cleaning extracted text...")
        cleaned_text = self.cleaner.clean(extracted.full_text)
        logger.info(
            f"  Cleaned: {extracted.total_chars:,} → {len(cleaned_text):,} characters "
            f"({(1 - len(cleaned_text)/extracted.total_chars)*100:.1f}% noise removed)"
        )

        # ---- Stage 3: Chunk ----
        logger.info("[3/5] Chunking with metadata extraction...")

        # Get metadata from law_config or infer from filename
        act_name = (law_config or {}).get("name", pdf_path.stem.replace("_", " ").title())
        act_id = (law_config or {}).get("id", pdf_path.stem)
        category = (law_config or {}).get("category", "")
        year = (law_config or {}).get("year", 0)
        language = (law_config or {}).get("language", "english")

        chunks = self.chunker.chunk_document(
            text=cleaned_text,
            act_name=act_name,
            act_id=act_id,
            category=category,
            year=year,
            language=language,
        )
        logger.info(f"  Created {len(chunks)} chunks")

        # Convert to dicts for export
        chunk_dicts = [chunk.to_dict() for chunk in chunks]

        # ---- Stage 4: Export ----
        logger.info("[4/5] Exporting processed data...")

        # Save as JSON (primary format for Phase 2)
        json_path = self.output_dir / f"{act_id}_chunks.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f, indent=2, ensure_ascii=False)
        logger.info(f"  JSON saved: {json_path}")

        # Save as CSV (for easy inspection in spreadsheets)
        csv_path = self.output_dir / f"{act_id}_chunks.csv"
        self._export_csv(chunk_dicts, csv_path)
        logger.info(f"  CSV saved: {csv_path}")

        # Save cleaned full text (useful for debugging)
        text_path = self.output_dir / f"{act_id}_cleaned.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        logger.info(f"  Cleaned text saved: {text_path}")

        # ---- Stage 5: Quality Report ----
        logger.info("[5/5] Generating quality report...")
        report = self.reporter.generate_report(
            chunks=chunk_dicts,
            act_name=act_name,
            output_path=self.output_dir / f"{act_id}_quality_report.json",
        )

        result = {
            "status": "success",
            "filename": pdf_path.name,
            "act_name": act_name,
            "act_id": act_id,
            "pages_extracted": extracted.total_pages,
            "chars_raw": extracted.total_chars,
            "chars_cleaned": len(cleaned_text),
            "total_chunks": len(chunks),
            "total_tokens": sum(c.token_count for c in chunks),
            "avg_tokens_per_chunk": round(
                sum(c.token_count for c in chunks) / max(len(chunks), 1), 1
            ),
            "quality_score": report["quality_score"]["score"],
            "quality_grade": report["quality_score"]["grade"],
            "output_files": {
                "json": str(json_path),
                "csv": str(csv_path),
                "cleaned_text": str(text_path),
                "quality_report": str(self.output_dir / f"{act_id}_quality_report.json"),
            },
        }

        logger.info(
            f"  ✓ Done: {len(chunks)} chunks, "
            f"Quality: {report['quality_score']['score']}/100 "
            f"({report['quality_score']['grade']})"
        )

        return result

    def process_all(
        self,
        priority_filter: str | None = None,
    ) -> dict:
        """
        Process all registered law PDFs.

        Args:
            priority_filter: Optional filter — "P0", "P1", or "P2".
                           If None, processes all available PDFs.

        Returns:
            Summary dictionary with all processing results.
        """
        logger.info("\n" + "=" * 60)
        logger.info("AinSeba Ingestion Pipeline — Batch Processing")
        logger.info("=" * 60)

        # Determine which laws to process
        if priority_filter:
            laws_to_process = [
                law for law in LAW_REGISTRY if law["priority"] == priority_filter
            ]
        else:
            laws_to_process = LAW_REGISTRY

        results = {
            "run_date": datetime.now().isoformat(),
            "total_laws": len(laws_to_process),
            "processed": [],
            "skipped": [],
            "failed": [],
        }

        all_chunks = []

        for law in laws_to_process:
            pdf_path = self.raw_dir / law["filename"]

            if not pdf_path.exists():
                logger.warning(f"⚠ PDF not found: {pdf_path}. Skipping.")
                results["skipped"].append({
                    "act_id": law["id"],
                    "act_name": law["name"],
                    "reason": f"PDF not found at {pdf_path}",
                })
                continue

            try:
                result = self.process_single_pdf(pdf_path, law_config=law)
                if result["status"] == "success":
                    results["processed"].append(result)

                    # Collect all chunks for combined export
                    json_path = result["output_files"]["json"]
                    with open(json_path, "r", encoding="utf-8") as f:
                        all_chunks.extend(json.load(f))
                else:
                    results["failed"].append(result)

            except Exception as e:
                logger.error(f"✗ Error processing {law['name']}: {e}")
                results["failed"].append({
                    "status": "error",
                    "act_id": law["id"],
                    "act_name": law["name"],
                    "error": str(e),
                })

        # Export combined chunks (all laws in one file)
        if all_chunks:
            combined_json = self.output_dir / "all_chunks_combined.json"
            with open(combined_json, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"\nCombined export: {len(all_chunks)} total chunks → {combined_json}")

            combined_csv = self.output_dir / "all_chunks_combined.csv"
            self._export_csv(all_chunks, combined_csv)

        # Generate combined quality report
        if all_chunks:
            combined_report = self.reporter.generate_report(
                chunks=all_chunks,
                act_name="All Documents Combined",
                output_path=self.output_dir / "combined_quality_report.json",
            )
            results["combined_quality"] = combined_report["quality_score"]

        # Summary
        results["summary"] = {
            "processed_count": len(results["processed"]),
            "skipped_count": len(results["skipped"]),
            "failed_count": len(results["failed"]),
            "total_chunks": len(all_chunks),
            "total_tokens": sum(c.get("token_count", 0) for c in all_chunks),
        }

        # Save pipeline run results
        run_log = self.output_dir / "pipeline_run_log.json"
        with open(run_log, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self._print_summary(results)

        return results

    def _export_csv(self, chunks: list[dict], csv_path: Path) -> None:
        """Export chunks to CSV for easy inspection."""
        if not chunks:
            return

        fieldnames = [
            "chunk_id", "act_name", "act_id", "part", "chapter",
            "section_number", "section_title", "category", "year",
            "language", "token_count", "chunk_index", "text",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for chunk in chunks:
                row = {k: chunk.get(k, "") for k in fieldnames}
                # Truncate text for CSV readability
                if len(str(row.get("text", ""))) > 500:
                    row["text"] = str(row["text"])[:500] + "..."
                # Convert list to string for CSV
                if isinstance(row.get("page_numbers"), list):
                    row["page_numbers"] = str(row["page_numbers"])
                writer.writerow(row)

    def _print_summary(self, results: dict) -> None:
        """Print a summary of the pipeline run."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel

            console = Console()

            console.print(Panel(
                "[bold green]Pipeline Run Complete[/bold green]",
                border_style="green",
            ))

            table = Table(title="Processing Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            s = results["summary"]
            table.add_row("Processed", str(s["processed_count"]))
            table.add_row("Skipped (PDF missing)", str(s["skipped_count"]))
            table.add_row("Failed", str(s["failed_count"]))
            table.add_row("Total Chunks", f"{s['total_chunks']:,}")
            table.add_row("Total Tokens", f"{s['total_tokens']:,}")

            if "combined_quality" in results:
                cq = results["combined_quality"]
                table.add_row("Quality Score", f"{cq['score']}/100 ({cq['grade']})")

            console.print(table)

            if results["skipped"]:
                console.print("\n[yellow]Skipped:[/yellow]")
                for s in results["skipped"]:
                    console.print(f"  ⚠ {s['act_name']}: {s['reason']}")

            if results["failed"]:
                console.print("\n[red]Failed:[/red]")
                for f in results["failed"]:
                    console.print(f"  ✗ {f.get('act_name', 'Unknown')}: {f.get('error', f.get('reason', ''))}")

        except ImportError:
            s = results["summary"]
            print(f"\n{'='*60}")
            print(f"Pipeline Complete")
            print(f"Processed: {s['processed_count']}, Skipped: {s['skipped_count']}, Failed: {s['failed_count']}")
            print(f"Total Chunks: {s['total_chunks']:,}, Total Tokens: {s['total_tokens']:,}")
            print(f"{'='*60}")
