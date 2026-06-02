"""Align stage: OpenAI segments source/target text into matching chunks."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import datasets as ds

from kotoba_benchmark._prompts import find_prompt
from kotoba_benchmark._vendored._align_manager import TextAlignmentManager
from kotoba_benchmark.config import Config


def _require_llm_api_key(model: str, *, stage: str) -> None:
    if model.lower().startswith("gemini-"):
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise RuntimeError(
                f"GEMINI_API_KEY (or GOOGLE_API_KEY) is required for the {stage} stage "
                f"(model={model!r})"
            )
    elif not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            f"OPENAI_API_KEY is required for the {stage} stage (model={model!r})"
        )

logger = logging.getLogger(__name__)


def _timestamps_to_align_input(timestamps: list[dict] | None) -> list[dict] | None:
    """Convert {start, end, text} entries to the alignment manager's expected format."""
    if not timestamps:
        return None
    out: list[dict] = []
    for ts in timestamps:
        if not isinstance(ts, dict):
            continue
        start = ts.get("start")
        end = ts.get("end")
        text = ts.get("text") if "text" in ts else ts.get("word")
        if start is None or end is None or not isinstance(text, str):
            continue
        try:
            out.append({"start": float(start), "end": float(end), "text": text})
        except (TypeError, ValueError):
            continue
    return out or None


def align_dataset(
    *, dataset: ds.Dataset, config: Config, output_dir: Path
) -> ds.Dataset:
    """Align source and target transcripts into parallel segment lists.

    Adds columns:
      - `aligned_source_segments` (list[str] | None per row)
      - `aligned_target_segments` (list[str] | None per row)
    """

    _require_llm_api_key(config.align.model, stage="align")

    source_col = f"timestamps_{config.source_lang}"
    target_col = f"timestamps_{config.target_lang}"
    for col in (source_col, target_col):
        if col not in dataset.column_names:
            raise KeyError(f"align stage: dataset missing column {col!r}")

    prompt_path = find_prompt(
        "align",
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        version=config.align.version,
    )
    logger.info("align: using prompt %s", prompt_path)

    cache_dir = output_dir / "_cache" / "align" / prompt_path.stem
    manager = TextAlignmentManager(
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        cache_dir=cache_dir if config.align.use_cache else None,
        use_cache=config.align.use_cache,
        max_workers=config.align.max_workers,
        max_retries=config.align.max_retries,
        model_name=config.align.model,
        rps_limit=config.align.rps_limit,
        request_timeout_sec=config.align.request_timeout_seconds,
        prompt_path=prompt_path,
        show_progress=config.show_progress(),
    )

    source_inputs: list[list[dict] | None] = [
        _timestamps_to_align_input(ts) for ts in dataset[source_col]
    ]
    target_inputs: list[list[dict] | None] = [
        _timestamps_to_align_input(ts) for ts in dataset[target_col]
    ]

    valid_indices = [
        i for i, (s, t) in enumerate(zip(source_inputs, target_inputs)) if s and t
    ]

    aligned_source: list[list[str] | None] = [None] * len(dataset)
    aligned_target: list[list[str] | None] = [None] * len(dataset)

    if valid_indices:
        batch_sources = [source_inputs[i] for i in valid_indices]
        batch_targets = [target_inputs[i] for i in valid_indices]
        out_sources, out_targets = manager.get_aligned_segments_in_parallel(
            batch_sources, batch_targets
        )
        for i, src, tgt in zip(valid_indices, out_sources, out_targets):
            if src and tgt:
                aligned_source[i] = list(src)
                aligned_target[i] = list(tgt)

    ok = sum(1 for x in aligned_source if x is not None)
    logger.info("align stage complete: %d/%d ok", ok, len(dataset))

    dataset = dataset.add_column("aligned_source_segments", aligned_source)
    dataset = dataset.add_column("aligned_target_segments", aligned_target)
    return dataset
