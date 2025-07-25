"""Central configuration module for the Multi‑Agent RAG project.

All runtime options live here so that code in `agents/`, `pipeline.py`, and
`evaluation.py` can do:

    from config import settings

and get a validated, immutable `Settings` object that merges:
1. **Defaults** defined below.
2. **Environment variables** (all with the prefix `RAG_`).
3. **Values in a local `.env` file** – handy for development.

Fail‑fast validators catch common mistakes (e.g. critic weights not summing to 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from pydantic import BaseSettings, Field, validator

# -----------------------------------------------------------------------------
# Optional: fine‑grained provider block (useful if you juggle many endpoints)
# -----------------------------------------------------------------------------


class ProviderCfg(BaseSettings):
    """Settings for a single LLM provider / endpoint."""

    name: str  # e.g. "openai", "vertex", "local_phibot"
    model_id: str  # full model path or name
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096

    class Config:
        extra = "forbid"  # reject unknown fields


# -----------------------------------------------------------------------------
# Main settings object used across the codebase
# -----------------------------------------------------------------------------


class Settings(BaseSettings):
    """Project‑wide configuration with sensible defaults and env overrides."""

    # === Authentication / project IDs ===
    openai_key: str = Field(..., env="RAG_OPENAI_KEY")
    vertex_project: str = Field("", env="RAG_VERTEX_PROJECT")

    # === Agent parameters ===
    max_tokens_summary: int = Field(4096, env="RAG_MAX_TOKENS_SUMMARY")
    critic_weights: Tuple[float, float, float] = Field(
        (0.5, 0.3, 0.2), env="RAG_CRITIC_WEIGHTS"
    )  # coverage, factuality, style

    # === Model ensemble ===
    model_list: List[str] = Field(
        default_factory=lambda: [
            "gpt-4o",  # OpenAI – ensure `openai_key` is set
            "projects/YOUR_PROJECT/locations/us-central1/models/medpalm2",  # Vertex
        ],
        env="RAG_MODEL_LIST",
    )

    # Alternatively define fully structured provider configs
    providers: List[ProviderCfg] = []

    # === Runtime paths & misc ===
    prompt_dir: Path = Field("prompts/", env="RAG_PROMPT_DIR")
    log_level: str = Field("INFO", env="RAG_LOG_LEVEL")
    cache_dir: Path = Field(".cache/", env="RAG_CACHE_DIR")

    # ------------------------------------------------------------------
    # Pydantic v1 style config – switch to `model_config` when on v2
    # ------------------------------------------------------------------

    class Config:
        env_file = ".env"
        env_prefix = "RAG_"  # applies to *all* fields unless explicit env=...
        case_sensitive = False

    # ------------------------------------------------------------------
    # Validators / computed properties
    # ------------------------------------------------------------------

    @validator("critic_weights")
    def _weights_must_sum_to_one(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if abs(sum(v) - 1.0) > 1e-3:
            raise ValueError("critic_weights must sum to 1.0")
        return v

    @property
    def main_provider(self) -> str:
        """Convenience accessor – returns the first provider or model string."""
        if self.providers:
            return self.providers[0].name
        return self.model_list[0]


# -----------------------------------------------------------------------------
# Public singleton – import this everywhere instead of instantiating repeatedly
# -----------------------------------------------------------------------------

settings = Settings()

__all__ = ["Settings", "settings", "ProviderCfg"]
