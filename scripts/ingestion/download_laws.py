"""
AinSeba - Law PDF Download Guide & Utility
Helps download Bangladesh law PDFs from the official government portal.

NOTE: bdlaws.minlaw.gov.bd serves HTML, not direct PDF downloads.
You'll need to save PDFs manually from the website or use a 
browser-based approach. This script provides guidance and validates
downloaded files.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LAW_REGISTRY, RAW_DATA_DIR


def print_download_guide():
    """Print instructions for downloading law PDFs."""
    print("=" * 70)
    print("AinSeba — Law PDF Download Guide")
    print("=" * 70)
    print()
    print("The official Bangladesh law portal (bdlaws.minlaw.gov.bd) provides")
    print("laws in HTML format. To get PDFs, follow these steps:")
    print()
    print("METHOD 1: Browser Print-to-PDF (Recommended)")
    print("-" * 50)
    print("1. Open the URL in Chrome/Edge")
    print("2. Press Ctrl+P (Print)")
    print("3. Set Destination to 'Save as PDF'")
    print("4. Save with the exact filename listed below")
    print()
    print("METHOD 2: Copy-Paste to Document")
    print("-" * 50)
    print("1. Open the URL and select all text (Ctrl+A)")
    print("2. Copy and paste into a Word/Google Doc")
    print("3. Export as PDF with the filename listed below")
    print()
    print("METHOD 3: Use sample/test PDFs first")
    print("-" * 50)
    print("To test the pipeline without downloading all laws,")
    print("you can create a small test PDF. Run:")
    print("  python scripts/ingestion/create_sample_pdf.py")
    print()
    print("=" * 70)
    print("REQUIRED PDFs (save to data/raw/ folder):")
    print("=" * 70)

    for i, law in enumerate(LAW_REGISTRY, 1):
        priority_color = {
            "P0": "🔴",
            "P1": "🟡",
            "P2": "🟢",
        }.get(law["priority"], "⚪")

        print(f"\n{i}. {priority_color} [{law['priority']}] {law['name']}")
        print(f"   Filename: {law['filename']}")
        print(f"   URL:      {law['source_url']}")
        print(f"   Category: {law['category']}")

    print(f"\nSave all PDFs to: {RAW_DATA_DIR.absolute()}")


def check_downloaded_pdfs():
    """Check which PDFs have been downloaded."""
    print("\n" + "=" * 70)
    print("Checking downloaded PDFs...")
    print("=" * 70)

    found = 0
    missing = 0

    for law in LAW_REGISTRY:
        pdf_path = RAW_DATA_DIR / law["filename"]
        if pdf_path.exists():
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {law['filename']} ({size_mb:.1f} MB)")
            found += 1
        else:
            print(f"  ✗ {law['filename']} — MISSING")
            missing += 1

    print(f"\nFound: {found}/{len(LAW_REGISTRY)} PDFs")
    if missing > 0:
        print(f"Missing: {missing} PDFs — download from URLs above")
    else:
        print("All PDFs downloaded! Ready to run the pipeline.")

    return found, missing


if __name__ == "__main__":
    print_download_guide()
    check_downloaded_pdfs()
