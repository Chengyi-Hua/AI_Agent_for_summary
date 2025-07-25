"""Asynchronous Summarizer Agent

Generates multiple draft summaries of a clinical case by combining diverse
prompt styles with multiple LLM providers configured in `config.py`.
Each summary is returned as a `SummaryDraft` dataclass that carries useful
metadata (model, prompt style, token estimate) for downstream agents.

The class is **async‑friendly**: all generation calls are dispatched in
parallel using `asyncio.gather`, which keeps overall latency low even when
hitting several external APIs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import jinja2
import openai

from config import settings

openai.api_key = settings.openai_key

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SummaryDraft:
    """A single candidate summary."""

    id: int
    content: str
    model: str
    prompt_style: str
    token_count: int


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class SummarizerAgent:
    """Fan‑out summarisation agent."""

    _DEFAULT_PROMPT_STYLES: Sequence[str] = (
        "soap",          # Subjective/Objective/Assessment/Plan
        "problem_list",  # Bullet list of problems
        "checklist",     # Key Dx / Tx / TODO checkout list
    )

    def __init__(
        self,
        providers: Sequence[str] | None = None,
        prompt_dir: str | Path | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.providers: Sequence[str] = providers or settings.model_list
        self.prompt_dir: Path = Path(prompt_dir or settings.prompt_dir)
        self.max_tokens: int = max_tokens or settings.max_tokens_summary

        # Jinja environment (file‑based if templates exist, else fallback)
        self._jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.prompt_dir),
            autoescape=False,
            keep_trailing_newline=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def summarise(
        self,
        case_text: str,
        prompt_styles: Sequence[str] | None = None,
    ) -> List[SummaryDraft]:
        """Return a list of draft summaries."""

        styles = prompt_styles or self._DEFAULT_PROMPT_STYLES
        tasks = []
        draft_id = 0

        for provider in self.providers:
            for style in styles:
                draft_id += 1
                tasks.append(
                    asyncio.create_task(
                        self._generate(
                            draft_id=draft_id,
                            provider=provider,
                            prompt_style=style,
                            case_text=case_text,
                        )
                    )
                )

        drafts = await asyncio.gather(*tasks)
        # filter out Nones when generation fails
        return [d for d in drafts if d is not None]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate(
        self,
        draft_id: int,
        provider: str,
        prompt_style: str,
        case_text: str,
    ) -> SummaryDraft | None:  # noqa: D401
        """Generate a draft summary via the specified provider + prompt."""

        prompt = self._render_prompt(prompt_style, case_text)

        try:
            resp = await openai.ChatCompletion.acreate(
                model=provider,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert clinical summarisation assistant "
                            "who writes concise, factual summaries."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=self.max_tokens,
            )
            content = resp.choices[0].message.content.strip()
            token_estimate = int(len(content.split()) * 0.75)  # rough heuristic
            return SummaryDraft(
                id=draft_id,
                content=content,
                model=provider,
                prompt_style=prompt_style,
                token_count=token_estimate,
            )
        except Exception as exc:  # noqa: BLE001,E722 – broad but logged
            print(
                f"[WARN] Summary generation failed (provider={provider}, "
                f"style={prompt_style}): {exc}"
            )
            return None

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _render_prompt(self, style: str, case_text: str) -> str:
        """Return a fully‑rendered prompt (file‑based or inline fallback)."""

        template_path = self.prompt_dir / f"{style}.j2"
        if template_path.exists():
            template = self._jinja_env.get_template(f"{style}.j2")
            return template.render(case_text=case_text)

        # -------- inline fallback templates --------
        if style == "soap":
            return (
                "Below is a de‑identified clinical case report. Write a concise "
                "\nSOAP‑style summary (≤ 200 words).\n\n---\n" + case_text
            )
        if style == "problem_list":
            return (
                "Read the following clinical text. Produce a bullet PROBLEM LIST "
                "(max 15 items, one per line).\n\n---\n" + case_text
            )
        if style == "checklist":
            return (
                "Summarise the case as a three‑section checklist containing (1) Key "
                "Diagnoses, (2) Therapies/Meds, (3) Outstanding Questions. Each "
                "section ≤ 4 bullets.\n\n---\n" + case_text
            )

        # default: vanilla summary prompt
        return "Summarise the following clinical case:\n\n" + case_text
