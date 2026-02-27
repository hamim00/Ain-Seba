#!/usr/bin/env python3
"""
AinSeba - Phase 4 Runner (Bilingual Support)
CLI entry point for bilingual legal queries.

Usage:
    python scripts/run_bilingual.py --query "শ্রমিকের সর্বোচ্চ কর্মঘণ্টা কত?"
    python scripts/run_bilingual.py --query "amar malik betan dey nai, ki korbo?"
    python scripts/run_bilingual.py --detect "আমার মালিক বেতন দেয় না"
    python scripts/run_bilingual.py --interactive
    python scripts/run_bilingual.py --test
    python scripts/run_bilingual.py --lang bn --query "What is the penalty for theft?"
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OPENAI_API_KEY, PROCESSED_DATA_DIR


def setup_logging(level: str = "WARNING"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def print_detection(detection):
    """Print language detection result."""
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()

        table = Table(title="Language Detection")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Language", detection.language.value)
        table.add_row("Confidence", f"{detection.confidence:.2f}")
        table.add_row("Bangla Ratio", f"{detection.bangla_ratio:.2f}")
        table.add_row("Has Bangla Script", str(detection.has_bangla_script))
        table.add_row("Has Latin Script", str(detection.has_latin_script))
        table.add_row("Is Mixed", str(detection.is_mixed))
        table.add_row("Needs Translation", str(detection.needs_translation))
        table.add_row("Response Language", detection.response_language)
        console.print(table)

    except ImportError:
        print(f"Language: {detection.language.value}")
        print(f"Confidence: {detection.confidence:.2f}")
        print(f"Needs Translation: {detection.needs_translation}")


def print_bilingual_response(response, show_sources: bool = True):
    """Pretty-print a bilingual response."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown

        console = Console()

        # Detection info
        lang_label = {
            "en": "English",
            "bn": "Bangla",
            "banglish": "Banglish",
        }.get(response.detected_language, response.detected_language)

        console.print(
            f"\n[dim]Detected: {lang_label} "
            f"(confidence: {response.detection_confidence:.0%}) | "
            f"Response: {response.response_language}[/dim]"
        )

        if response.was_translated:
            console.print(
                f"[dim]Translated query: {response.query_english[:80]}...[/dim]"
            )

        # Answer
        console.print(Panel(
            Markdown(response.answer),
            title="[bold green]AinSeba Response[/bold green]",
            border_style="green",
        ))

        # English version (if response is in Bangla)
        if response.response_language == "bn" and response.answer != response.answer_english:
            console.print(Panel(
                Markdown(response.answer_english),
                title="[bold blue]English Version[/bold blue]",
                border_style="blue",
            ))

        # Sources
        if show_sources and response.sources:
            console.print(f"\n[bold]Sources ({len(response.sources)}):[/bold]")
            for i, src in enumerate(response.sources, 1):
                console.print(f"  [cyan][{i}][/cyan] {src['citation']}")

    except ImportError:
        print(f"\nDetected: {response.detected_language}")
        print(f"Translated: {response.was_translated}")
        print(f"\n{'='*70}")
        print(response.answer)
        if show_sources and response.sources:
            print(f"\nSources:")
            for i, src in enumerate(response.sources, 1):
                print(f"  [{i}] {src['citation']}")
        print(f"{'='*70}")


def cmd_detect(text: str):
    """Detect language only."""
    from src.language.detector import detect_language
    detection = detect_language(text)
    print(f"\nText: {text}")
    print_detection(detection)


def cmd_query(question: str, response_lang: str = None, no_reranker: bool = False):
    """Run a bilingual query."""
    from src.chain.builder import build_bilingual_chain

    chain = build_bilingual_chain(
        use_reranker=not no_reranker,
        response_language=response_lang or "auto",
    )
    response = chain.query(question, response_language=response_lang)
    print_bilingual_response(response)
    return response


def cmd_interactive(no_reranker: bool = False):
    """Interactive bilingual chat mode."""
    from src.chain.builder import build_bilingual_chain

    chain = build_bilingual_chain(use_reranker=not no_reranker)
    session_id = f"bilingual_{datetime.now().strftime('%H%M%S')}"

    print(f"\n{'='*70}")
    print("AinSeba - Bilingual Legal Assistant")
    print("Ask questions in English, Bangla, or Banglish.")
    print("Commands: /lang en|bn, /detect, /clear, /quit  (or just type: quit, exit, q)")
    print(f"{'='*70}")

    force_lang = None

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! / বিদায়!")
            break

        if not question:
            continue

        if question in ("/quit", "quit", "exit", "q", "bye", "/exit"):
            print("Goodbye! / বিদায়!")
            break
        elif question.startswith("/lang "):
            lang = question.split(" ", 1)[1].strip()
            if lang in ("en", "bn", "auto"):
                force_lang = lang if lang != "auto" else None
                print(f"Response language set to: {lang}")
            else:
                print("Use: /lang en, /lang bn, or /lang auto")
            continue
        elif question == "/detect":
            print("Type text to detect language:")
            text = input("  > ").strip()
            if text:
                cmd_detect(text)
            continue
        elif question == "/clear":
            chain.rag_chain.clear_conversation(session_id)
            print("Conversation memory cleared.")
            continue

        response = chain.query(
            question=question,
            session_id=session_id,
            response_language=force_lang,
        )
        print_bilingual_response(response)


def cmd_test(no_reranker: bool = False):
    """Run bilingual test queries."""
    from src.language.detector import detect_language

    # Test language detection first (no API needed)
    detection_tests = [
        # (text, expected_language)
        ("What is the penalty for theft?", "en"),
        ("চুরির শাস্তি কী?", "bn"),
        ("আমার মালিক ৩ মাস বেতন দেয়নি", "bn"),
        ("amar malik betan dey nai ki korbo", "banglish"),
        ("ami jante chai ain ki bole", "banglish"),
        ("Section 42 অনুযায়ী working hours কত?", "bn"),
        ("What are শ্রমিকের rights?", "en"),
        ("Hello, how are you?", "en"),
        ("কনজিউমার রাইটস কি?", "bn"),
        ("korte parbo ki ami case file", "banglish"),
    ]

    print(f"\n{'='*70}")
    print("AinSeba - Bilingual Detection Test")
    print(f"{'='*70}")

    passed = 0
    for text, expected in detection_tests:
        result = detect_language(text)
        status = "[OK]" if result.language.value == expected else "[FAIL]"
        if result.language.value == expected:
            passed += 1
        print(
            f"  {status} '{text[:50]}...' "
            f"-> {result.language.value} "
            f"(expected: {expected}, conf: {result.confidence:.2f})"
        )

    print(f"\nDetection: {passed}/{len(detection_tests)} passed")

    # Test full bilingual pipeline (requires API key)
    if not OPENAI_API_KEY:
        print("\nSkipping full pipeline tests (no OPENAI_API_KEY)")
        return

    from src.chain.builder import build_bilingual_chain
    chain = build_bilingual_chain(use_reranker=not no_reranker)

    bilingual_queries = [
        {
            "query": "What are the maximum working hours per day?",
            "type": "english",
            "expected_lang": "en",
        },
        {
            "query": "শ্রমিকের সর্বোচ্চ কর্মঘণ্টা কত?",
            "type": "bangla",
            "expected_lang": "bn",
        },
        {
            "query": "amar malik betan dey nai, ki korbo?",
            "type": "banglish",
            "expected_lang": "bn",
        },
    ]

    print(f"\n{'='*70}")
    print("AinSeba - Full Bilingual Pipeline Test")
    print(f"{'='*70}")

    all_results = []
    for i, tq in enumerate(bilingual_queries, 1):
        print(f"\n--- Test {i}: [{tq['type']}] ---")
        print(f"Q: {tq['query']}")

        try:
            response = chain.query(question=tq["query"])

            print(f"Detected: {response.detected_language}")
            print(f"Translated: {response.was_translated}")
            if response.was_translated:
                print(f"English query: {response.query_english[:80]}")
            print(f"Response lang: {response.response_language}")
            print(f"Answer: {response.answer[:200]}...")
            print(f"Sources: {len(response.sources)}")

            all_results.append(response.to_dict())

        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({"query": tq["query"], "error": str(e)})

    # Save results
    output_path = PROCESSED_DATA_DIR / "phase4_bilingual_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "detection_passed": passed,
            "detection_total": len(detection_tests),
            "pipeline_results": all_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AinSeba - Phase 4: Bilingual Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_bilingual.py --detect "আমার মালিক বেতন দেয় না"
  python scripts/run_bilingual.py --query "চুরির শাস্তি কী?"
  python scripts/run_bilingual.py --query "amar malik betan dey nai"
  python scripts/run_bilingual.py --lang bn --query "penalty for theft"
  python scripts/run_bilingual.py --interactive
  python scripts/run_bilingual.py --test
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="Ask a question (any language)")
    group.add_argument("--detect", type=str, help="Detect language only")
    group.add_argument("--interactive", action="store_true", help="Interactive bilingual chat")
    group.add_argument("--test", action="store_true", help="Run bilingual test suite")

    parser.add_argument("--lang", type=str, choices=["en", "bn", "auto"], default=None,
                       help="Force response language")
    parser.add_argument("--no-reranker", action="store_true", help="Skip reranking")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING"])

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.detect:
        cmd_detect(args.detect)
    elif args.query:
        if not OPENAI_API_KEY:
            print("ERROR: OPENAI_API_KEY not set.")
            sys.exit(1)
        cmd_query(args.query, response_lang=args.lang, no_reranker=args.no_reranker)
    elif args.interactive:
        if not OPENAI_API_KEY:
            print("ERROR: OPENAI_API_KEY not set.")
            sys.exit(1)
        cmd_interactive(no_reranker=args.no_reranker)
    elif args.test:
        cmd_test(no_reranker=args.no_reranker)


if __name__ == "__main__":
    main()
