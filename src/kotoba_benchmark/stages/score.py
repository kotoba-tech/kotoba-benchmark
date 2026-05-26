"""Score stage: per-chunk OpenAI scoring + latency metrics."""

from __future__ import annotations

import logging
from pathlib import Path

import datasets as ds

from kotoba_benchmark._prompts import find_prompt
from kotoba_benchmark._vendored._chunk_timestamps import ChunkTimestampsProcessor
from kotoba_benchmark._vendored._eval_manager import TranslationEvaluationManager
from kotoba_benchmark.config import Config

logger = logging.getLogger(__name__)


def _aggregate_chunk_timestamps(
    text_chunks: list[str] | None,
    timestamps: list[dict] | None,
) -> list[dict] | None:
    """Map aligned text chunks back to per-chunk {start, end} timestamps.

    Returns one timestamp dict per chunk, or None if mapping is impossible.
    """
    if not text_chunks or not timestamps:
        return None
    processor = ChunkTimestampsProcessor(text_key="text")
    try:
        aggregated = processor.aggregate_timestamps(text_chunks, timestamps)
    except Exception as exc:  # noqa: BLE001 — coverage mismatch / non-text chunks
        logger.debug("timestamp aggregation failed: %s", exc)
        return None
    return [
        {"start": float(s), "end": float(e), "text": text}
        for (s, e), text in zip(aggregated, text_chunks)
    ]


def _collect_row_latency(
    outputs: list[dict] | None,
    source_ts: list[dict] | None,
    target_ts: list[dict] | None,
) -> list[float]:
    """Start-latency per chunk where accuracy==1 and timestamps are valid."""
    if not outputs or not source_ts or not target_ts:
        return []
    latencies: list[float] = []
    for out, src, tgt in zip(outputs, source_ts, target_ts):
        if not isinstance(out, dict):
            continue
        accuracy = out.get("accuracy")
        if not isinstance(accuracy, (int, float)) or float(accuracy) != 1.0:
            continue
        src_start = src.get("start") if isinstance(src, dict) else None
        tgt_start = tgt.get("start") if isinstance(tgt, dict) else None
        src_end = src.get("end") if isinstance(src, dict) else None
        tgt_end = tgt.get("end") if isinstance(tgt, dict) else None
        if not all(isinstance(x, (int, float)) for x in (src_start, tgt_start, src_end, tgt_end)):
            continue
        start_latency = tgt_start - src_start
        end_latency = tgt_end - src_end
        if start_latency >= 0 and end_latency >= 0:
            latencies.append(float(start_latency))
    return latencies


def score_dataset(*, dataset: ds.Dataset, config: Config, output_dir: Path) -> ds.Dataset:
    """Score aligned chunks via OpenAI evaluation + compute latency metrics.

    Adds columns:
      - `output` (list[dict] | None per row): per-chunk {accuracy, fluency, conciseness, ...}
      - `chunked_timestamps_<src>`, `chunked_timestamps_<tgt>` (list[dict] | None per row)
      - `latencies` (list[float] per row): per-chunk start-latencies, accuracy==1 only
    """

    from kotoba_benchmark.stages.align import _require_llm_api_key
    _require_llm_api_key(config.evaluate.model, stage="score")

    for col in ("aligned_source_segments", "aligned_target_segments"):
        if col not in dataset.column_names:
            raise KeyError(f"score stage: dataset missing column {col!r}")

    prompt_path = find_prompt(
        "evaluate",
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        version=config.evaluate.version,
    )
    logger.info("evaluate: using prompt %s", prompt_path)

    cache_dir = output_dir / "_cache" / "evaluate" / prompt_path.stem
    manager = TranslationEvaluationManager(
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        cache_dir=cache_dir if config.evaluate.use_cache else None,
        use_cache=config.evaluate.use_cache,
        max_workers=config.evaluate.max_workers,
        max_retries=config.evaluate.max_retries,
        model_name=config.evaluate.model,
        rps_limit=config.evaluate.rps_limit,
        request_timeout_sec=config.evaluate.request_timeout_seconds,
        prompt_path=prompt_path,
    )

    aligned_source = dataset["aligned_source_segments"]
    aligned_target = dataset["aligned_target_segments"]

    valid_indices = [
        i for i, (s, t) in enumerate(zip(aligned_source, aligned_target)) if s and t
    ]

    outputs: list[list[dict] | None] = [None] * len(dataset)
    if valid_indices:
        batch_sources = [aligned_source[i] for i in valid_indices]
        batch_targets = [aligned_target[i] for i in valid_indices]
        batch_outputs = manager.evaluate_translations_in_parallel(
            batch_sources, batch_targets
        )
        for i, out in zip(valid_indices, batch_outputs):
            outputs[i] = out

    source_ts_col = f"timestamps_{config.source_lang}"
    target_ts_col = f"timestamps_{config.target_lang}"
    source_ts_all = dataset[source_ts_col]
    target_ts_all = dataset[target_ts_col]

    chunked_source: list[list[dict] | None] = []
    chunked_target: list[list[dict] | None] = []
    latencies: list[list[float]] = []

    for i in range(len(dataset)):
        src_chunked = _aggregate_chunk_timestamps(aligned_source[i], source_ts_all[i])
        tgt_chunked = _aggregate_chunk_timestamps(aligned_target[i], target_ts_all[i])
        chunked_source.append(src_chunked)
        chunked_target.append(tgt_chunked)
        latencies.append(_collect_row_latency(outputs[i], src_chunked, tgt_chunked))

    n_scored = sum(1 for o in outputs if o is not None)
    logger.info("score stage complete: %d/%d rows scored", n_scored, len(dataset))

    dataset = dataset.add_column("output", outputs)
    dataset = dataset.add_column(f"chunked_timestamps_{config.source_lang}", chunked_source)
    dataset = dataset.add_column(f"chunked_timestamps_{config.target_lang}", chunked_target)
    dataset = dataset.add_column("latencies", latencies)
    return dataset
