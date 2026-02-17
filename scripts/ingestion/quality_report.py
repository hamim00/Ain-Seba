"""
AinSeba - Chunk Quality Report Generator
Inspects processed chunks and generates a quality assessment report.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class QualityReporter:
    """
    Generates quality reports for processed document chunks.
    
    Checks:
    - Token count distribution (are chunks within target range?)
    - Metadata completeness (does every chunk have act_name, section, etc.?)
    - Content quality (empty chunks, very short chunks, duplicates)
    - Sample chunks for manual inspection
    """

    def __init__(
        self,
        target_min_tokens: int = 100,
        target_max_tokens: int = 800,
    ):
        self.target_min_tokens = target_min_tokens
        self.target_max_tokens = target_max_tokens

    def generate_report(
        self,
        chunks: list[dict],
        act_name: str = "All Documents",
        output_path: str | Path | None = None,
    ) -> dict:
        """
        Generate a quality report for a list of chunks.

        Args:
            chunks: List of chunk dictionaries (from Chunk.to_dict()).
            act_name: Name of the law being reported on.
            output_path: Optional path to save the report as JSON.

        Returns:
            Report dictionary with stats and issues.
        """
        if not chunks:
            return {"error": "No chunks to analyze"}

        report = {
            "report_date": datetime.now().isoformat(),
            "document": act_name,
            "summary": self._compute_summary(chunks),
            "token_distribution": self._analyze_token_distribution(chunks),
            "metadata_completeness": self._check_metadata_completeness(chunks),
            "content_issues": self._find_content_issues(chunks),
            "sample_chunks": self._get_sample_chunks(chunks),
        }

        # Overall quality score (simple heuristic)
        report["quality_score"] = self._compute_quality_score(report)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Quality report saved to: {output_path}")

        return report

    def _compute_summary(self, chunks: list[dict]) -> dict:
        """Basic statistics about the chunks."""
        token_counts = [c.get("token_count", 0) for c in chunks]
        unique_sections = set()
        unique_chapters = set()

        for c in chunks:
            if c.get("section_number"):
                unique_sections.add(c["section_number"])
            if c.get("chapter"):
                unique_chapters.add(c["chapter"])

        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": round(sum(token_counts) / len(token_counts), 1),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "median_tokens": sorted(token_counts)[len(token_counts) // 2],
            "unique_sections": len(unique_sections),
            "unique_chapters": len(unique_chapters),
        }

    def _analyze_token_distribution(self, chunks: list[dict]) -> dict:
        """Analyze how well chunks fit the target token range."""
        token_counts = [c.get("token_count", 0) for c in chunks]

        in_range = sum(
            1 for t in token_counts
            if self.target_min_tokens <= t <= self.target_max_tokens
        )
        too_small = sum(1 for t in token_counts if t < self.target_min_tokens)
        too_large = sum(1 for t in token_counts if t > self.target_max_tokens)

        # Buckets for histogram
        buckets = {
            "0-100": 0,
            "100-200": 0,
            "200-400": 0,
            "400-600": 0,
            "600-800": 0,
            "800+": 0,
        }
        for t in token_counts:
            if t < 100:
                buckets["0-100"] += 1
            elif t < 200:
                buckets["100-200"] += 1
            elif t < 400:
                buckets["200-400"] += 1
            elif t < 600:
                buckets["400-600"] += 1
            elif t < 800:
                buckets["600-800"] += 1
            else:
                buckets["800+"] += 1

        return {
            "target_range": f"{self.target_min_tokens}-{self.target_max_tokens} tokens",
            "in_range_count": in_range,
            "in_range_pct": round(in_range / len(chunks) * 100, 1),
            "too_small_count": too_small,
            "too_large_count": too_large,
            "distribution_buckets": buckets,
        }

    def _check_metadata_completeness(self, chunks: list[dict]) -> dict:
        """Check what percentage of chunks have each metadata field."""
        fields = ["act_name", "act_id", "chapter", "section_number", "section_title", "category"]
        completeness = {}

        for field in fields:
            filled = sum(1 for c in chunks if c.get(field))
            completeness[field] = {
                "filled": filled,
                "total": len(chunks),
                "pct": round(filled / len(chunks) * 100, 1),
            }

        return completeness

    def _find_content_issues(self, chunks: list[dict]) -> list[dict]:
        """Identify potential quality issues."""
        issues = []

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            tokens = chunk.get("token_count", 0)

            # Very short chunks
            if tokens < 30:
                issues.append({
                    "type": "very_short",
                    "chunk_id": chunk.get("chunk_id", f"idx_{i}"),
                    "token_count": tokens,
                    "preview": text[:100],
                })

            # Very long chunks
            if tokens > self.target_max_tokens * 1.5:
                issues.append({
                    "type": "very_long",
                    "chunk_id": chunk.get("chunk_id", f"idx_{i}"),
                    "token_count": tokens,
                    "preview": text[:100],
                })

            # Missing section info
            if not chunk.get("section_number") and not chunk.get("section_title"):
                issues.append({
                    "type": "missing_section_info",
                    "chunk_id": chunk.get("chunk_id", f"idx_{i}"),
                    "preview": text[:100],
                })

            # Chunks that look like they're just headers
            if tokens < 20 and any(
                kw in text.upper() for kw in ["CHAPTER", "PART", "SCHEDULE"]
            ):
                issues.append({
                    "type": "header_only",
                    "chunk_id": chunk.get("chunk_id", f"idx_{i}"),
                    "preview": text[:100],
                })

        return issues

    def _get_sample_chunks(self, chunks: list[dict], n: int = 5) -> list[dict]:
        """Get a representative sample of chunks for manual inspection."""
        if len(chunks) <= n:
            sample_indices = list(range(len(chunks)))
        else:
            # Pick evenly spaced samples
            step = len(chunks) // n
            sample_indices = [i * step for i in range(n)]

        samples = []
        for idx in sample_indices:
            chunk = chunks[idx]
            samples.append({
                "chunk_index": idx,
                "chunk_id": chunk.get("chunk_id", ""),
                "section_number": chunk.get("section_number", ""),
                "section_title": chunk.get("section_title", ""),
                "chapter": chunk.get("chapter", ""),
                "token_count": chunk.get("token_count", 0),
                "text_preview": chunk.get("text", "")[:300] + "...",
            })

        return samples

    def _compute_quality_score(self, report: dict) -> dict:
        """
        Compute an overall quality score (0-100).
        Based on: token distribution, metadata completeness, issues.
        """
        score = 100
        reasons = []

        # Token distribution (40 points)
        in_range_pct = report["token_distribution"]["in_range_pct"]
        token_score = min(40, in_range_pct * 0.4)
        score = score - (40 - token_score)
        if in_range_pct < 70:
            reasons.append(f"Only {in_range_pct}% chunks in target token range")

        # Metadata completeness (30 points)
        meta = report["metadata_completeness"]
        section_pct = meta.get("section_number", {}).get("pct", 0)
        meta_score = min(30, section_pct * 0.3)
        score = score - (30 - meta_score)
        if section_pct < 70:
            reasons.append(f"Only {section_pct}% chunks have section numbers")

        # Issues penalty (30 points)
        issues = report["content_issues"]
        total_chunks = report["summary"]["total_chunks"]
        issue_rate = len(issues) / total_chunks if total_chunks > 0 else 0
        issue_penalty = min(30, issue_rate * 100)
        score = score - issue_penalty
        if issue_rate > 0.1:
            reasons.append(f"{len(issues)} quality issues found ({issue_rate*100:.0f}% of chunks)")

        return {
            "score": round(max(0, score), 1),
            "grade": self._score_to_grade(score),
            "improvement_suggestions": reasons,
        }

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 90:
            return "A - Excellent"
        elif score >= 80:
            return "B - Good"
        elif score >= 70:
            return "C - Acceptable"
        elif score >= 60:
            return "D - Needs Improvement"
        else:
            return "F - Poor"


def print_report_summary(report: dict) -> None:
    """Print a human-readable report summary to the console."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        # Header
        console.print(Panel(
            f"[bold]Quality Report: {report['document']}[/bold]\n"
            f"Generated: {report['report_date']}",
            title="AinSeba Chunk Quality Report",
            border_style="blue",
        ))

        # Summary stats
        summary = report["summary"]
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total Chunks: {summary['total_chunks']}")
        console.print(f"  Total Tokens: {summary['total_tokens']:,}")
        console.print(f"  Avg Tokens/Chunk: {summary['avg_tokens_per_chunk']}")
        console.print(f"  Token Range: {summary['min_tokens']} - {summary['max_tokens']}")
        console.print(f"  Unique Sections: {summary['unique_sections']}")
        console.print(f"  Unique Chapters: {summary['unique_chapters']}")

        # Token distribution
        dist = report["token_distribution"]
        console.print(f"\n[bold]Token Distribution:[/bold]")
        console.print(f"  In target range: {dist['in_range_count']} ({dist['in_range_pct']}%)")
        console.print(f"  Too small: {dist['too_small_count']}")
        console.print(f"  Too large: {dist['too_large_count']}")

        # Distribution histogram
        table = Table(title="Token Distribution")
        table.add_column("Bucket", style="cyan")
        table.add_column("Count", style="green")
        for bucket, count in dist["distribution_buckets"].items():
            bar = "█" * min(count, 50)
            table.add_row(bucket, f"{count} {bar}")
        console.print(table)

        # Quality score
        qs = report["quality_score"]
        score_color = "green" if qs["score"] >= 80 else "yellow" if qs["score"] >= 60 else "red"
        console.print(f"\n[bold]Quality Score:[/bold] [{score_color}]{qs['score']}/100 - {qs['grade']}[/{score_color}]")

        if qs["improvement_suggestions"]:
            console.print("\n[bold]Suggestions:[/bold]")
            for suggestion in qs["improvement_suggestions"]:
                console.print(f"  ⚠ {suggestion}")

        # Issues
        issues = report["content_issues"]
        if issues:
            console.print(f"\n[bold]Issues Found: {len(issues)}[/bold]")
            for issue in issues[:10]:
                console.print(f"  [{issue['type']}] {issue['chunk_id']}: {issue.get('preview', '')[:80]}")
            if len(issues) > 10:
                console.print(f"  ... and {len(issues) - 10} more")

    except ImportError:
        # Fallback without rich
        print(f"\n{'='*60}")
        print(f"Quality Report: {report['document']}")
        print(f"{'='*60}")
        summary = report["summary"]
        print(f"Total Chunks: {summary['total_chunks']}")
        print(f"Avg Tokens: {summary['avg_tokens_per_chunk']}")
        qs = report["quality_score"]
        print(f"Quality Score: {qs['score']}/100 - {qs['grade']}")
        print(f"{'='*60}")
