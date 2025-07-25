"""
agents/keyword_extractor.py
===========================

Keyword / MeSH Extractor
-----------------------
Takes the *winning* clinical summary and returns a **JSON-serialisable list**
of PubMed-ready keywords (ideally MeSH terms).  It calls an LLM to do the
concept spotting, then applies tiny post-processing to canonicalise obvious
variants (whitespace, plural “s”, capitalisation).

The module is completely self-contained and asynchronous, so it fits the rest
of the pipeline.

Example
~~~~~~~
>>> extractor = KeywordExtractor()
>>> terms = await extractor.extract(summary_text)
>>> print(terms)
['myocardial infarction', 'angioplasty', 'aspirin therapy', ...]
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import List

import openai

from config import settings

openai.api_key = settings.openai_key


# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #
class KeywordExtractor:
    """
    LLM-backed keyword extractor.

    Parameters
    ----------
    model : str | None
        Which LLM to hit.  Defaults to the *first* provider in ``settings``.
    max_attempts : int
        How many times to retry if the model fails to return valid JSON.
    """

    def __init__(self, model: str | None = None, *, max_attempts: int = 3) -> None:
        self.model = (
            model
            or (settings.providers[0]["model"] if isinstance(settings.providers[0], dict) else settings.providers[0])
        )
        self.max_attempts = max_attempts

    # --------------------------------------------------------------------- #
    #  external                                                             #
    # --------------------------------------------------------------------- #
    async def extract(self, summary: str, *, max_terms: int = 8) -> List[str]:
        """
        Return up to ``max_terms`` PubMed-style keywords for the given summary.

        The result is **guaranteed** to be valid Python list of *unique*,
        lowercase strings.
        """
        for attempt in range(1, self.max_attempts + 1):
            try:
                prompt = self._build_prompt(summary, max_terms)
                resp = await openai.ChatCompletion.acreate(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=128,
                    timeout=20,
                )
                raw_json = resp.choices[0].message.content.strip()
                terms: List[str] = json.loads(raw_json)

                # Defensive normalisation
                terms = self._normalise_terms(terms)
                return terms[:max_terms]

            except (openai.error.OpenAIError, json.JSONDecodeError, ValueError):
                if attempt == self.max_attempts:
                    return []  # Fail-gracefully: caller can decide what to do
                await asyncio.sleep(0.5 * attempt)

    # --------------------------------------------------------------------- #
    #  helpers                                                              #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _build_prompt(summary: str, max_terms: int) -> str:
        return (
            "You are a biomedical indexing assistant.\n\n"
            "From the clinical summary delimited by <summary></summary>, extract **up to "
            f"{max_terms} PubMed-ready keywords or MeSH descriptors** that best represent the case.\n"
            "Return your answer **as a JSON array of lowercase strings** ONLY—no prose, no extra keys.\n\n"
            "<summary>\n"
            f"{summary}\n"
            "</summary>"
        )

    @staticmethod
    def _normalise_terms(terms: List[str]) -> List[str]:
        """
        Lower-case, strip punctuation/extra-spaces, singularise trivial plurals,
        and de-duplicate while preserving order.
        """
        seen, cleaned = set(), []
        for term in terms:
            tok = term.lower().strip()
            tok = re.sub(r"[.;,]+", "", tok)            # drop punctuation
            tok = re.sub(r"\s{2,}", " ", tok)           # collapse spaces
            tok = re.sub(r"s$", "", tok) if len(tok) > 4 else tok  # crude plural→singular

            if tok and tok not in seen:
                seen.add(tok)
                cleaned.append(tok)
        return cleaned


# --------------------------------------------------------------------------- #
#  smoke-test:  python -m agents.keyword_extractor                            #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    async def _demo() -> None:
        EXAMPLE = (
            "Concise summary: 62-year-old male with acute ST-elevation myocardial infarction "
            "treated with primary PCI to the LAD, started on dual antiplatelet therapy and "
            "high-dose statin. Past history includes hypertension and type-2 diabetes."
        )
        print("⏳  Querying LLM…")
        terms = await KeywordExtractor().extract(EXAMPLE)
        print("✅  Keywords:", terms)

    asyncio.run(_demo())
