"""Transcribe stage: Gemini-based source + target transcription with timestamps."""

from __future__ import annotations

import logging

import datasets as ds

from kotoba_benchmark._vendored._gemini_transcribe import transcribe_dataset_with_gemini
from kotoba_benchmark.config import Config

logger = logging.getLogger(__name__)


def transcribe_dataset(*, dataset: ds.Dataset, config: Config) -> ds.Dataset:
    """Add `timestamps_<source_lang>` and `timestamps_<target_lang>` columns.

    Each column is `list[{"start": float, "end": float, "text": str}]` per row,
    produced by Gemini structured output. Empty list on failure.
    """

    source_col = f"audio_{config.source_lang}"
    target_col = f"audio_{config.target_lang}"

    for col in (source_col, target_col):
        if col not in dataset.column_names:
            raise KeyError(f"transcribe stage: dataset missing column {col!r}")

    tc = config.transcribe

    logger.info("transcribing source audio (%s)...", config.source_lang)
    dataset = transcribe_dataset_with_gemini(
        input_dataset=dataset,
        input_audio_column=source_col,
        input_lang=config.source_lang,
        model=tc.model,
        temperature=tc.temperature,
        max_output_tokens=tc.max_output_tokens,
        thinking_level=tc.thinking_level,
        request_timeout_seconds=tc.request_timeout_seconds,
        max_retries=tc.max_retries,
        max_workers=tc.max_workers,
        show_progress=config.show_progress(),
    )

    logger.info("transcribing target audio (%s)...", config.target_lang)
    dataset = transcribe_dataset_with_gemini(
        input_dataset=dataset,
        input_audio_column=target_col,
        input_lang=config.target_lang,
        model=tc.model,
        temperature=tc.temperature,
        max_output_tokens=tc.max_output_tokens,
        thinking_level=tc.thinking_level,
        request_timeout_seconds=tc.request_timeout_seconds,
        max_retries=tc.max_retries,
        max_workers=tc.max_workers,
        show_progress=config.show_progress(),
    )

    return dataset
