#!/usr/bin/env python3
"""
AinSeba - Phase 3 Runner (RAG Chain & LLM)
CLI entry point for querying the legal assistant.

Usage:
    python scripts/run_chain.py --query "What is the penalty for theft?"
    python scripts/run_chain.py --interactive
    python scripts/run_chain.py --stream "My employer hasn't paid me"
    python scripts/run_chain.py --test
    python scripts/run_chain.py --context "working hours"    # Preview retrieval only
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


def print_response(response, show_sources: bool = True):
    """Pretty-print a RAG response."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown

        console = Console()

        # Answer
        console.print(Panel(
            Markdown(response.answer),
            title="[bold green]AinSeba Response[/bold green]",
            border_style="green",
        ))

        # Sources
        if show_sources and response.sources:
            console.print(f"\n[bold]Sources ({len(response.sources)}):[/bold]")
            for i, src in enumerate(response.sources, 1):
                score_info = f"sim={src['similarity_score']:.3f}"
                if src.get('rerank_score'):
                    score_info += f", rerank={src['rerank_score']:.3f}"
                console.print(
                    f"  [cyan][{i}][/cyan] {src['citation']} "
                    f"[dim]({score_info})[/dim]"
                )

    except ImportError:
        print(f"\n{'='*70}")
        print("AinSeba Response")
        print(f"{'='*70}")
        print(response.answer)
        if show_sources and response.sources:
            print(f"\nSources ({len(response.sources)}):")
            for i, src in enumerate(response.sources, 1):
                print(f"  [{i}] {src['citation']}")
        print(f"{'='*70}")


def cmd_query(question: str, no_reranker: bool = False):
    """Run a single query."""
    from src.chain.builder import build_rag_chain

    chain = build_rag_chain(use_reranker=not no_reranker)
    response = chain.query(question)
    print_response(response)
    return response


def cmd_stream(question: str, no_reranker: bool = False):
    """Run a query with streaming output."""
    from src.chain.builder import build_rag_chain

    chain = build_rag_chain(use_reranker=not no_reranker)

    print(f"\nQ: {question}\n")
    print("-" * 50)

    for token in chain.stream(question, use_reranker=not no_reranker):
        print(token, end="", flush=True)

    print("\n" + "-" * 50)


def cmd_interactive(no_reranker: bool = False):
    """Interactive chat mode with conversation memory."""
    from src.chain.builder import build_rag_chain

    chain = build_rag_chain(use_reranker=not no_reranker)
    session_id = f"interactive_{datetime.now().strftime('%H%M%S')}"

    print(f"\n{'='*70}")
    print("AinSeba - Interactive Legal Assistant")
    print("Ask legal questions about Bangladesh law.")
    print("Commands: /clear (reset memory), /history, /sources, /quit")
    print(f"{'='*70}")

    last_response = None

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question == "/quit":
            print("Goodbye!")
            break
        elif question == "/clear":
            chain.clear_conversation(session_id)
            print("Conversation memory cleared.")
            continue
        elif question == "/history":
            history = chain.get_conversation_history(session_id)
            if not history:
                print("No conversation history yet.")
            else:
                for msg in history:
                    role = "You" if msg["role"] == "user" else "AinSeba"
                    content = msg["content"][:150]
                    print(f"  {role}: {content}...")
            continue
        elif question == "/sources":
            if last_response and last_response.sources:
                print(f"\nSources from last response:")
                for i, src in enumerate(last_response.sources, 1):
                    print(f"  [{i}] {src['citation']}")
                    print(f"      Preview: {src.get('text_preview', '')[:100]}...")
            else:
                print("No sources available.")
            continue

        # Query the chain
        response = chain.query(
            question=question,
            session_id=session_id,
            use_reranker=not no_reranker,
        )
        last_response = response

        print_response(response, show_sources=True)


def cmd_context_preview(question: str, no_reranker: bool = False):
    """Preview retrieved context without calling the LLM."""
    from src.chain.builder import build_rag_chain

    chain = build_rag_chain(use_reranker=not no_reranker)
    context = chain.get_context_preview(question)

    print(f"\nQuery: {question}")
    print(f"\n{'='*70}")
    print("Retrieved Context (what the LLM would see):")
    print(f"{'='*70}")
    print(context)


def cmd_test(no_reranker: bool = False):
    """Run Phase 3 Q&A tests with diverse query types."""
    from src.chain.builder import build_rag_chain

    chain = build_rag_chain(use_reranker=not no_reranker)

    test_queries = [
        # Direct factual
        {
            "query": "What are the maximum working hours per day for a worker?",
            "type": "direct",
        },
        # Situational
        {
            "query": "My employer hasn't paid me for 3 months. What can I do according to the law?",
            "type": "situational",
        },
        # Section-specific
        {
            "query": "What does Section 7 say about deductions from wages?",
            "type": "section_specific",
        },
        # Follow-up (tests memory)
        {
            "query": "What penalties apply if my employer violates that section?",
            "type": "follow_up",
        },
        # Out of scope
        {
            "query": "What is the tax rate in the United States?",
            "type": "out_of_scope",
        },
    ]

    session_id = "test_session"
    all_results = []

    print(f"\n{'='*70}")
    print("AinSeba - Phase 3 Q&A Test")
    print(f"Running {len(test_queries)} test queries")
    print(f"{'='*70}")

    for i, tq in enumerate(test_queries, 1):
        print(f"\n--- Test {i}/{len(test_queries)}: [{tq['type']}] ---")
        print(f"Q: {tq['query']}")

        try:
            response = chain.query(
                question=tq["query"],
                session_id=session_id,
                use_reranker=not no_reranker,
            )

            print(f"\nA: {response.answer[:300]}...")
            print(f"\nSources: {len(response.sources)}")
            for src in response.sources[:3]:
                print(f"  - {src['citation']}")

            all_results.append({
                "query": tq["query"],
                "type": tq["type"],
                "answer": response.answer,
                "sources": response.sources,
                "retrieval_count": response.retrieval_count,
            })

        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                "query": tq["query"],
                "type": tq["type"],
                "error": str(e),
            })

    # Save results
    output_path = PROCESSED_DATA_DIR / "phase3_qa_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "model": chain.model_name,
            "total_queries": len(test_queries),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Test complete! Results saved to: {output_path}")
    print(f"{'='*70}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="AinSeba - Phase 3: RAG Chain & LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_chain.py --query "What is the penalty for theft?"
  python scripts/run_chain.py --stream "worker rights in Bangladesh"
  python scripts/run_chain.py --interactive
  python scripts/run_chain.py --context "working hours"
  python scripts/run_chain.py --test
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="Ask a single question")
    group.add_argument("--stream", type=str, help="Ask with streaming output")
    group.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    group.add_argument("--context", type=str, help="Preview retrieved context (no LLM call)")
    group.add_argument("--test", action="store_true", help="Run Q&A test suite")

    parser.add_argument("--no-reranker", action="store_true", help="Skip cross-encoder reranking")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING"])

    args = parser.parse_args()
    setup_logging(args.log_level)

    # Check API key
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set.")
        print("  Create a .env file with: OPENAI_API_KEY=sk-your-key-here")
        sys.exit(1)

    if args.query:
        cmd_query(args.query, no_reranker=args.no_reranker)
    elif args.stream:
        cmd_stream(args.stream, no_reranker=args.no_reranker)
    elif args.interactive:
        cmd_interactive(no_reranker=args.no_reranker)
    elif args.context:
        cmd_context_preview(args.context, no_reranker=args.no_reranker)
    elif args.test:
        cmd_test(no_reranker=args.no_reranker)


if __name__ == "__main__":
    main()
