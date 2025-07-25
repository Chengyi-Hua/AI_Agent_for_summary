"""
evaluation.py
=============

Batch-evaluation harness for the multi-agent summarisation pipeline.

Usage
-----
$ python evaluation.py --data data/pilot_cases.jsonl \\
                       --outfile results/pilot_metrics.csv \\
                       --limit 50                   # optional truncation
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from bert_score import score as bert_score
from rouge import Rouge

# --- internal pipeline imports -------------------------------------------
from agents.summarizer_agent import SummarizerAgent
from agents.critic_agent import CriticAgent
from agents.voter import Voter

# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------
rouge = Rouge()  # computes ROUGE-1 and ROUGE-L by default


async def _run_single(case: Dict) -> Dict:
    """Process ONE case dict and return metrics + winner summary."""
    text = case["full_text"]
    gold = case["reference_summary"]

    # 1) multi-agent summary generation
    drafts = await SummarizerAgent().summarise(text)
    critiques = await CriticAgent().critique(drafts)
    winner = Voter(method="weighted").pick_best(critiques)

    # 2) automatic metrics
    rouge_scores = rouge.get_scores(winner.draft.content, gold, avg=True)
    (p, r, f1), *_ = bert_score(
        [winner.draft.content], [gold], model_type="roberta-large-mnli", verbose=False
    )

    return {
        "case_id": case.get("case_id", ""),
        "summary": winner.draft.content,
        "rouge_1": rouge_scores["rouge-1"]["f"],
        "rouge_l": rouge_scores["rouge-l"]["f"],
        "bert_f1": float(f1),
    }


async def evaluate(dataset_path: Path, limit: int | None = None) -> List[Dict]:
    """Evaluate the whole dataset asynchronously in batches."""
    cases = [json.loads(line) for line in dataset_path.open("r", encoding="utf-8")]
    if limit:
        cases = cases[:limit]

    # Process 5 cases at a time to avoid rate-limits
    chunk_size = 5
    results: List[Dict] = []
    for i in range(0, len(cases), chunk_size):
        batch = cases[i : i + chunk_size]
        coro = [_run_single(c) for c in batch]
        results.extend(await asyncio.gather(*coro))

    return results


# -------------------------------------------------------------------------
# CLI entry-point
# -------------------------------------------------------------------------
def _save_csv(rows: List[Dict], outfile: Path) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["case_id", "rouge_1", "rouge_l", "bert_f1", "summary"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch evaluation")
    parser.add_argument("--data", required=True, type=Path, help="JSONL dataset")
    parser.add_argument("--outfile", required=True, type=Path, help="CSV results path")
    parser.add_argument("--limit", type=int, default=None, help="process first N cases")
    args = parser.parse_args()

    rows = asyncio.run(evaluate(args.data, args.limit))

    # aggregate metrics
    avg_r1 = mean(r["rouge_1"] for r in rows)
    avg_rl = mean(r["rouge_l"] for r in rows)
    avg_bf = mean(r["bert_f1"] for r in rows)

    print(f"\nFinished {len(rows)} cases")
    print(f"ROUGE-1  : {avg_r1:.4f}")
    print(f"ROUGE-L  : {avg_rl:.4f}")
    print(f"BERTScore: {avg_bf:.4f}")

    _save_csv(rows, args.outfile)
    print(f"Per-case details written to {args.outfile}")


if __name__ == "__main__":
    main()
