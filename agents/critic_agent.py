"""agents/critic_agent.py
================================

Asynchronous **Critic Agent** that grades candidate summaries on three axes:

1. **Coverage**   – Does the draft include all clinically salient facts?
2. **Factuality** – Are those facts consistent with the source text?
3. **Readability** (Style) – Is the writing concise and clear?

It returns a list of `Critique` dataclasses that carry the per‑dimension
scores (1‑5) *and* a weighted total score using the weights specified in
`config.settings.critic_weights`.

Key design details
------------------
* Uses **asyncio.gather** so all LLM grading calls run in parallel.
* Implements **exponential‑back‑off retry** (max 3 attempts) for network or
  JSON‑parsing failures.
* Clamps scores to 1‑5 so a rogue LLM reply cannot crash downstream logic.
* Falls back to a *minimal* score if every retry fails (so the voter can still
  ignore that draft rather than the whole pipeline aborting).

Usage example
-------------
```python
from agents.summarizer_agent import SummarizerAgent
from agents.critic_agent import CriticAgent

summarizer = SummarizerAgent()
critic     = CriticAgent()

drafts     = await summarizer.summarise(case_text)
critiques  = await critic.critique(drafts)
```
"""

from __future__ import annotations

import asyncio
import json
import math
import random
from dataclasses import dataclass
from typing import List

import openai

from config import settings
from agents.summarizer_agent import SummaryDraft

openai.api_key = settings.openai_key  # picked up from .env or RAG_OPENAI_KEY

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Critique:
    """Container for per‑dimension scores plus the originating draft."""

    draft: SummaryDraft
    coverage: int
    factuality: int
    style: int
    weighted_total: float

    def as_dict(self) -> dict:
        return {
            "draft_id": id(self.draft),
            "model": self.draft.model,
            "prompt_style": self.draft.prompt_style,
            "coverage": self.coverage,
            "factuality": self.factuality,
            "style": self.style,
            "weighted_total": self.weighted_total,
        }

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _clamp(v: int) -> int:
        """Ensure a score is within 1‑5 inclusive."""
        return max(1, min(5, v))

    @classmethod
    def from_json(cls, raw_json: str, draft: SummaryDraft) -> "Critique":
        """Parses JSON produced by the LLM and performs basic validation."""
        data = json.loads(raw_json)
        cov  = cls._clamp(int(data.get("coverage", 1)))
        fact = cls._clamp(int(data.get("factuality", 1)))
        sty  = cls._clamp(int(data.get("style", 1)))
        w0, w1, w2 = settings.critic_weights
        total = cov * w0 + fact * w1 + sty * w2
        return cls(draft, cov, fact, sty, total)


# ---------------------------------------------------------------------------
# Critic Agent implementation
# ---------------------------------------------------------------------------

class CriticAgent:
    """LLM‑backed scorer for candidate summaries."""

    def __init__(self, model: str | None = None, *, max_retries: int = 3):
        self.model = (
            model
            or (settings.providers[0]["model"] if isinstance(settings.providers[0], dict) else settings.providers[0])
        )
        self.max_retries = max_retries

    # -------------------------- public API ---------------------------------

    async def critique(self, drafts: List[SummaryDraft]) -> List[Critique]:
        """Return a `Critique` for every draft (in parallel)."""
        tasks = [asyncio.create_task(self._critique_one(d)) for d in drafts]
        return await asyncio.gather(*tasks)

    # ------------------------- internal helpers ----------------------------

    async def _critique_one(self, draft: SummaryDraft) -> Critique:
        """Score a single draft with retry & error handling."""
        backoff = 1.5  # seconds
        for attempt in range(1, self.max_retries + 1):
            try:
                prompt = self._build_prompt(draft)
                resp = await openai.ChatCompletion.acreate(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=256,
                    timeout=30,
                )
                raw_json = resp.choices[0].message.content.strip()
                return Critique.from_json(raw_json, draft)

            except (openai.error.OpenAIError, json.JSONDecodeError, KeyError, ValueError) as exc:
                if attempt == self.max_retries:
                    # Give the draft a minimal score so the pipeline can continue
                    return Critique(
                        draft=draft,
                        coverage=1,
                        factuality=1,
                        style=1,
                        weighted_total=sum(settings.critic_weights),
                    )
                # jittered exponential back‑off
                await asyncio.sleep(backoff + random.uniform(0, 0.5))
                backoff *= 2

    # ------------------------- prompt builder ------------------------------

    @staticmethod
    def _build_prompt(draft: SummaryDraft) -> str:
        """Create the rubric prompt sent to the LLM."""
        w0, w1, w2 = settings.critic_weights
        reference_snippet = (
            draft.reference[:1500]
            if getattr(draft, "reference", None)
            else "(no reference available)"
        )
        return (
            "You are a senior clinical documentation reviewer. Evaluate the **candidate summary** "
            "below on three axes and reply **ONLY** with valid JSON in the following format:\n"
            "{\n  \"coverage\": <1‑5>,\n  \"factuality\": <1‑5>,\n  \"style\": <1‑5>,\n  \"weighted_total\": <float>\n}\n\n"
            f"The weighted_total must be computed as: coverage*{w0} + factuality*{w1} + style*{w2}.\n\n"
            "=== SOURCE REFERENCE (may be truncated) ===\n"
            f"{reference_snippet}\n\n"
            "=== CANDIDATE SUMMARY ===\n"
            f"{draft.content}"
        )
