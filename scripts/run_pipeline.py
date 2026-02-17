#!/usr/bin/env python3
"""
AinSeba - Pipeline Runner (CLI Entry Point)
Run the full ingestion pipeline from the command line.

Usage:
    python scripts/run_pipeline.py --sample          # Test with sample PDF
    python scripts/run_pipeline.py --all             # Process all available PDFs
    python scripts/run_pipeline.py --priority P0     # Process only P0 priority PDFs
    python scripts/run_pipeline.py --file path.pdf   # Process a single PDF
    python scripts/run_pipeline.py --check            # Check which PDFs are downloaded
    python scripts/run_pipeline.py --report           # View quality reports
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LAW_REGISTRY


def setup_logging(level: str = "INFO"):
    """Configure logging with clean formatting."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(PROCESSED_DATA_DIR / "pipeline.log", mode="a"),
        ],
    )


def run_sample():
    """Create a sample PDF and process it through the pipeline."""
    from scripts.ingestion.create_sample_pdf import create_sample_pdf, SAMPLE_LAW_CONFIG
    from scripts.ingestion.pipeline import IngestionPipeline
    from scripts.ingestion.quality_report import print_report_summary

    print("\n🧪 Running pipeline with SAMPLE PDF...\n")

    # Step 1: Create sample PDF
    pdf_path = create_sample_pdf()

    # Step 2: Run pipeline
    pipeline = IngestionPipeline()
    result = pipeline.process_single_pdf(pdf_path, law_config=SAMPLE_LAW_CONFIG)

    # Step 3: Show quality report
    if result["status"] == "success":
        report_path = result["output_files"]["quality_report"]
        with open(report_path, "r") as f:
            report = json.load(f)
        print_report_summary(report)

        print("\n" + "=" * 60)
        print("✅ Sample pipeline run COMPLETE!")
        print("=" * 60)
        print(f"\nOutput files:")
        for name, path in result["output_files"].items():
            print(f"  {name}: {path}")
        print(f"\n📊 Quality Score: {result['quality_score']}/100 ({result['quality_grade']})")
        print(f"📦 Total Chunks: {result['total_chunks']}")
        print(f"🔤 Total Tokens: {result['total_tokens']:,}")
    else:
        print(f"\n❌ Pipeline failed: {result.get('reason', 'Unknown error')}")

    return result


def run_all(priority_filter: str | None = None):
    """Process all available PDFs."""
    from scripts.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    results = pipeline.process_all(priority_filter=priority_filter)

    print("\n" + "=" * 60)
    if priority_filter:
        print(f"✅ Pipeline complete for priority: {priority_filter}")
    else:
        print("✅ Pipeline complete for all available PDFs")
    print("=" * 60)

    return results


def run_single(pdf_path: str):
    """Process a single PDF file."""
    from scripts.ingestion.pipeline import IngestionPipeline
    from scripts.ingestion.quality_report import print_report_summary

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    # Try to match with law registry
    law_config = None
    for law in LAW_REGISTRY:
        if law["filename"] == pdf_path.name:
            law_config = law
            break

    pipeline = IngestionPipeline()
    result = pipeline.process_single_pdf(pdf_path, law_config=law_config)

    if result["status"] == "success":
        report_path = result["output_files"]["quality_report"]
        with open(report_path, "r") as f:
            report = json.load(f)
        print_report_summary(report)

    return result


def check_pdfs():
    """Check which PDFs have been downloaded."""
    from scripts.ingestion.download_laws import print_download_guide, check_downloaded_pdfs

    print_download_guide()
    found, missing = check_downloaded_pdfs()
    return found, missing


def view_reports():
    """View existing quality reports."""
    from scripts.ingestion.quality_report import print_report_summary

    report_files = list(PROCESSED_DATA_DIR.glob("*_quality_report.json"))

    if not report_files:
        print("No quality reports found. Run the pipeline first.")
        return

    print(f"\nFound {len(report_files)} quality report(s):\n")

    for report_file in sorted(report_files):
        with open(report_file, "r") as f:
            report = json.load(f)
        print_report_summary(report)
        print()


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AinSeba - Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py --sample          Test with sample PDF
  python scripts/run_pipeline.py --all             Process all PDFs
  python scripts/run_pipeline.py --priority P0     Process P0 priority only
  python scripts/run_pipeline.py --file law.pdf    Process one PDF
  python scripts/run_pipeline.py --check           Check downloaded PDFs
  python scripts/run_pipeline.py --report          View quality reports
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true", help="Run with sample test PDF")
    group.add_argument("--all", action="store_true", help="Process all available PDFs")
    group.add_argument("--priority", type=str, choices=["P0", "P1", "P2"], help="Process by priority")
    group.add_argument("--file", type=str, help="Process a single PDF file")
    group.add_argument("--check", action="store_true", help="Check downloaded PDFs")
    group.add_argument("--report", action="store_true", help="View quality reports")

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Setup
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level)

    # Route to the right function
    if args.sample:
        run_sample()
    elif args.all:
        run_all()
    elif args.priority:
        run_all(priority_filter=args.priority)
    elif args.file:
        run_single(args.file)
    elif args.check:
        check_pdfs()
    elif args.report:
        view_reports()


if __name__ == "__main__":
    main()
