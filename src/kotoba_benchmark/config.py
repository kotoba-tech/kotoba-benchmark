"""Run configuration — the single source of truth for an evaluation.

Configs are TOML files or Python objects. CLI flags and the Python API both
materialize a `Config`; downstream stages take this object directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


class TranslateConfig(BaseModel):
    """Translate stage (STS). Default backend talks to a Kotoba endpoint via kotoba-sdk."""

    model_config = ConfigDict(extra="allow")  # backend-specific options pass through

    backend: str = "kotoba-sdk"
    url: str | None = None
    """Override the endpoint URL. When None, kotoba-sdk reads KOTOBA_S2ST_<SRC>_<TGT>_URL."""

    max_concurrency: int = 4
    max_retries: int = 3
    retry_interval_seconds: float = 5.0
    sample_rate: int = 24000
    chunk_ms: int = 40
    label: str | None = None
    """Free-form label for output naming (e.g. "partner_en2ja_staging")."""


class TranscribeConfig(BaseModel):
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    max_output_tokens: int = 8192 * 16
    thinking_level: str | None = None
    request_timeout_seconds: float = 300.0
    max_retries: int = 3
    max_workers: int = 32


class AlignConfig(BaseModel):
    model: str = "gemini-2.5-flash"
    """LLM for chunk alignment. Default `gemini-*` model routes through the native
    google-genai SDK; any `gpt-*` (or other) model routes through the openai SDK."""
    version: str | None = None
    """Pin a specific prompt version (e.g. "v1.7"). When None, latest pair-specific wins."""
    max_workers: int = 32
    max_retries: int = 3
    rps_limit: int = 50
    request_timeout_seconds: int = 300
    use_cache: bool = True


class EvaluateConfig(BaseModel):
    model: str = "gemini-2.5-flash"
    """LLM for chunk scoring. Same dispatch rule as `align.model`."""
    version: str | None = None
    max_workers: int = 32
    max_retries: int = 3
    rps_limit: int = 50
    request_timeout_seconds: int = 300
    use_cache: bool = True


class Config(BaseModel):
    """Top-level run config."""

    model_config = ConfigDict(extra="forbid")

    # --- inputs ---
    wav_dir: str | Path | None = None
    dataset: str | Path | None = None
    """HF dataset name or local save_to_disk path. Mutually exclusive with wav_dir."""

    source_lang: str
    target_lang: str

    # --- output ---
    output_dir: str | Path = Field(default_factory=lambda: Path("./out"))

    # --- stages ---
    translate: TranslateConfig = Field(default_factory=TranslateConfig)
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    align: AlignConfig = Field(default_factory=AlignConfig)
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)

    # --- summary ---
    write_summary: Literal["json", "json+md", "json+md+html", "none"] = "json+md+html"

    # --- in-memory backend instance (Python API only; not serializable to TOML) ---
    translate_backend: Any | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _check_inputs(self) -> Config:
        if self.wav_dir is None and self.dataset is None:
            raise ValueError("Config requires either wav_dir or dataset")
        if self.wav_dir is not None and self.dataset is not None:
            raise ValueError("Config takes wav_dir OR dataset, not both")
        return self

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        return cls(**data)

    def dataset_tag(self) -> str:
        """Stable identifier used in output filenames."""
        if self.translate.label:
            base = self.translate.label
        elif self.wav_dir is not None:
            base = Path(self.wav_dir).name
        else:
            base = Path(str(self.dataset)).name
        return f"{base}__{self.source_lang}2{self.target_lang}"
