"""
agents/voter.py
===============

Chooses the *best* summary draft after the CriticAgent has produced a
``Critique`` for every candidate.

Supported ranking strategies
----------------------------
* ``method="weighted"`` (default) – sort by ``Critique.weighted_total`` with a
  deterministic tie-breaking chain:
  coverage → factuality → style → model → prompt_style → draft id.

* ``method="borda"`` – classical Borda count: on each metric **individually**
  (coverage, factuality, style) the best draft gets *n* points, the next gets
  *n-1*, … then points are summed across metrics.

* ``method="irv"`` – Instant-Runoff (a.k.a. Ranked-Choice): repeatedly
  eliminates the draft with the fewest 1st-place votes (by weighted_total)
  until a single winner remains.  Useful if you ever need majority support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from agents.critic_agent import Critique


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
class Voter:
    """Aggregate Critique scores and pick a winner."""

    def __init__(self, *, method: str = "weighted") -> None:
        if method not in {"weighted", "borda", "irv"}:
            raise ValueError("method must be 'weighted', 'borda', or 'irv'")
        self.method = method

    # ---------------------------------------------------------------------
    # External helpers
    # ---------------------------------------------------------------------
    def rank(self, critiques: List[Critique]) -> List[Critique]:
        """Return *all* critiques in best-first order."""
        if not critiques:
            return []

        if self.method == "weighted":
            ranked = self._rank_weighted(critiques)
        elif self.method == "borda":
            ranked = self._rank_borda(critiques)
        else:  # 'irv'
            ranked = self._rank_irv(critiques)

        return ranked

    def pick_best(self, critiques: List[Critique]) -> Critique | None:
        """Return the single top-ranked Critique or None if list is empty."""
        return self.rank(critiques)[0] if critiques else None

    # ---------------------------------------------------------------------
    # Strategy: weighted sort
    # ---------------------------------------------------------------------
    @staticmethod
    def _rank_weighted(critiques: List[Critique]) -> List[Critique]:
        def tie_key(c: Critique) -> Tuple:
            return (
                -c.coverage,
                -c.factuality,
                -c.style,
                c.draft.model,
                c.draft.prompt_style,
                id(c.draft),
            )

        # primary key: negative weighted_total  (higher is better)
        return sorted(critiques, key=lambda c: (-c.weighted_total, *tie_key(c)))

    # ---------------------------------------------------------------------
    # Strategy: classical Borda count
    # ---------------------------------------------------------------------
    @staticmethod
    def _rank_borda(critiques: List[Critique]) -> List[Critique]:
        n = len(critiques)
        # rank each metric independently: higher value ⇒ better rank
        points: Dict[int, int] = {id(c): 0 for c in critiques}

        for metric in ("coverage", "factuality", "style"):
            sorted_metric = sorted(critiques, key=lambda c: getattr(c, metric), reverse=True)
            for position, crit in enumerate(sorted_metric):
                points[id(crit)] += n - position  # n points for best, n-1 next, …

        # Build final ordering; reuse the deterministic tie_key to break ties
        def tie_key(c: Critique) -> Tuple:
            return (
                -c.coverage,
                -c.factuality,
                -c.style,
                c.draft.model,
                c.draft.prompt_style,
                id(c.draft),
            )

        return sorted(critiques, key=lambda c: (-points[id(c)], *tie_key(c)))

    # ---------------------------------------------------------------------
    # Strategy: Instant-Runoff Voting (IRV)
    # ---------------------------------------------------------------------
    @staticmethod
    def _rank_irv(critiques: List[Critique]) -> List[Critique]:
        # Copy list so we can mutate
        remaining = critiques[:]

        # Helper: best-to-worst order by weighted score
        def order(lst: List[Critique]) -> List[Critique]:
            return sorted(lst, key=lambda c: (-c.weighted_total, -c.coverage, -c.factuality, -c.style))

        elimination_order: List[Critique] = []

        while len(remaining) > 1:
            ordered = order(remaining)
            # Tally 1st-place votes (weighted_total acts as the ranking ballot)
            top_score = ordered[0].weighted_total
            votes: Dict[Critique, int] = {}
            for crit in remaining:
                if crit.weighted_total == top_score:
                    votes[crit] = votes.get(crit, 0) + 1

            # Find the draft with *fewest* first-place votes
            loser = min(votes, key=votes.get)
            elimination_order.append(loser)
            remaining = [c for c in remaining if c is not loser]

        # remaining now has one (the winner); prepend elimination_order reversed
        return remaining + elimination_order[::-1]


# -----------------------------------------------------------------------------
# Ad-hoc smoke-test (python -m agents.voter)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Build three fake critiques for demonstration
    @dataclass
    class _FakeDraft:
        content: str
        model: str
        prompt_style: str

    fake_critiques = [
        Critique(draft=_FakeDraft("one", "gpt-4o", "soap"), coverage=5, factuality=4, style=4, weighted_total=4.5),
        Critique(draft=_FakeDraft("two", "gpt-4o", "check"), coverage=4, factuality=4, style=5, weighted_total=4.4),
        Critique(draft=_FakeDraft("three", "medpalm2", "problem"), coverage=4, factuality=5, style=4, weighted_total=4.55),
    ]

    for m in ("weighted", "borda", "irv"):
        winner = Voter(method=m).pick_best(fake_critiques)
        print(f"{m:8s} → {winner.draft.content!r}  score={winner.weighted_total:.2f}")
