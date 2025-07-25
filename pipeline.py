"""
pipeline.py
===========

Single-case entry-point for the multi-agent summarisation pipeline.

It

1.  reads the *full* clinical note (either from a file or stdin),
2.  generates multiple candidate summaries,
3.  scores them, picks the best one,
4.  extracts PubMed-ready keywords, and
5.  prints a JSON payload you can feed to the downstream PubMed retriever.

Usage
-----

# from a file
$ python pipeline.py --file examples/case_001.txt --out out/case_001.json

# or from stdin
$ cat examples/case_001.txt | python pipeline.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict

from agents.summarizer_agent import SummarizerAgent
from agents.critic_agent import CriticAgent, Critique
from agents.voter import Voter
from agents.keyword_extractor import KeywordExtractor


# -------------------------------------------------------------------------
# Core async routine
# -------------------------------------------------------------------------
async def run_pipeline(text: str) -> Dict:
    """Return a JSON-serialisable dict with summary + keywords + scores."""
    # 1. candidate generation
    drafts = await SummarizerAgent().summarise(text)

    # 2. scoring
    critiques: list[Critique] = await CriticAgent().critique(drafts)
    winner: Critique | None = Voter(method="weighted").pick_best(critiques)
    if winner is None:
        raise RuntimeError("No viable summary produced")

    # 3. keyword extraction
    keywords = await KeywordExtractor().extract(winner.draft.content)

    # 4. shape payload
    return {
        "summary": winner.draft.content,
        "model": winner.draft.model,
        "prompt_style": winner.draft.prompt_style,
        "scores": {
            "coverage": winner.coverage,
            "factuality": winner.factuality,
            "style": winner.style,
            "weighted_total": winner.weighted_total,
        },
        "keywords": keywords,
    }


# -------------------------------------------------------------------------
# CLI wrapper
# -------------------------------------------------------------------------
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-agent pipeline on one case")
    p.add_argument(
        "--file",
        type=Path,
        help="Path to text file containing the full clinical note. "
        "If omitted, pipeline reads from stdin.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON result to this path instead of stdout.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_cli()

    # ingest the case text
    if args.file:
        text = args.file.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    # run async pipeline
    result = asyncio.run(run_pipeline(text))

    # emit
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"✅  Result written to {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
