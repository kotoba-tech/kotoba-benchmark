"""End-to-end evaluation pipeline: translate → transcribe → align → score → report."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets as ds

# Importing the backend submodules triggers their @register decorators.
import kotoba_benchmark.stages.translate.kotoba_sdk  # noqa: F401
import kotoba_benchmark.stages.translate.openai_realtime  # noqa: F401
from kotoba_benchmark._audio import dataset_from_wav_dir
from kotoba_benchmark.config import Config
from kotoba_benchmark.report import build_summary, write_summary
from kotoba_benchmark.stages.align import align_dataset
from kotoba_benchmark.stages.score import score_dataset
from kotoba_benchmark.stages.transcribe import transcribe_dataset
from kotoba_benchmark.stages.translate._runner import (
    translate_dataset_async,
)

logger = logging.getLogger(__name__)

_PAIR_RE = re.compile(r"(?:^|__)([a-z]{2,3})2([a-z]{2,3})(?:__|$)")


@dataclass
class Result:
    """Evaluation result returned to callers."""

    tag: str
    output_dir: Path
    dataset: ds.Dataset
    summary: dict[str, Any]
    summary_paths: dict[str, Path]
    config: Config

    @property
    def scores(self) -> dict[str, Any]:
        """Chunk-level scores: accuracy, fluency, conciseness, latency."""
        return self.summary["metrics"]

    def write_summary(self, output_dir: str | Path) -> dict[str, Path]:
        """Re-render summary files to `output_dir` (writes JSON/MD/HTML)."""
        from kotoba_benchmark.report import write_summary as _write

        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        paths = _write(dataset=self.dataset, config=self.config, output_dir=target)
        self.summary_paths = paths
        return paths


def _load_input_dataset(config: Config) -> ds.Dataset:
    if config.wav_dir is not None:
        return dataset_from_wav_dir(config.wav_dir, source_lang=config.source_lang)
    dataset_spec = str(config.dataset)
    p = Path(dataset_spec).expanduser()
    if p.exists() and p.is_dir():
        return ds.load_from_disk(str(p))
    return ds.load_dataset(dataset_spec, split="train")


def _stage_cache_path(output_dir: Path, stage: str, tag: str) -> Path:
    return output_dir / f"_stage_cache__{stage}__{tag}"


def _load_stage_cache(path: Path) -> ds.Dataset | None:
    if path.is_dir():
        try:
            return ds.load_from_disk(str(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load stage cache %s: %s", path, exc)
    return None


def _save_stage_cache(dataset: ds.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataset.save_to_disk(str(path))
    except Exception as exc:  # noqa: BLE001 — caching is best-effort
        logger.warning("failed to save stage cache %s: %s", path, exc)


async def evaluate_async(config: Config) -> Result:
    """Run the full pipeline and return a Result."""

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = config.dataset_tag()
    logger.info("kotoba-benchmark: tag=%s output_dir=%s", tag, output_dir)

    # --- stage 1: translate ---
    translate_cache = _stage_cache_path(output_dir, "translate", tag)
    dataset = _load_stage_cache(translate_cache)
    if dataset is None:
        logger.info("[1/4] translate: running")
        dataset = _load_input_dataset(config)
        dataset = await translate_dataset_async(
            dataset=dataset, config=config, output_dir=output_dir
        )
        _save_stage_cache(dataset, translate_cache)
    else:
        logger.info("[1/4] translate: using cached %s", translate_cache.name)

    # --- stage 2: transcribe ---
    transcribe_cache = _stage_cache_path(output_dir, "transcribe", tag)
    cached = _load_stage_cache(transcribe_cache)
    if cached is None:
        logger.info("[2/4] transcribe: running")
        dataset = transcribe_dataset(dataset=dataset, config=config)
        _save_stage_cache(dataset, transcribe_cache)
    else:
        logger.info("[2/4] transcribe: using cached %s", transcribe_cache.name)
        dataset = cached

    # --- stage 3: align ---
    align_cache = _stage_cache_path(output_dir, "align", tag)
    cached = _load_stage_cache(align_cache)
    if cached is None:
        logger.info("[3/4] align: running")
        dataset = align_dataset(dataset=dataset, config=config, output_dir=output_dir)
        _save_stage_cache(dataset, align_cache)
    else:
        logger.info("[3/4] align: using cached %s", align_cache.name)
        dataset = cached

    # --- stage 4: score ---
    score_cache = _stage_cache_path(output_dir, "score", tag)
    cached = _load_stage_cache(score_cache)
    if cached is None:
        logger.info("[4/4] score: running")
        dataset = score_dataset(dataset=dataset, config=config, output_dir=output_dir)
        _save_stage_cache(dataset, score_cache)
    else:
        logger.info("[4/4] score: using cached %s", score_cache.name)
        dataset = cached

    summary_paths = write_summary(dataset=dataset, config=config, output_dir=output_dir)
    summary = build_summary(dataset=dataset, config=config, output_dir=output_dir)

    return Result(
        tag=tag,
        output_dir=output_dir,
        dataset=dataset,
        summary=summary,
        summary_paths=summary_paths,
        config=config,
    )


def evaluate(config: Config) -> Result:
    """Synchronous run of `evaluate_async`."""
    return asyncio.run(evaluate_async(config))


def re_render_summary(output_dir: str | Path) -> dict[str, Path]:
    """Re-render summary files from the last `score` stage cache in `output_dir`.

    Useful when the partner wants an updated HTML without re-running the pipeline.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    score_caches = sorted(output_dir.glob("_stage_cache__score__*"))
    if not score_caches:
        raise FileNotFoundError(f"No score-stage cache found under {output_dir}")
    cache = score_caches[-1]
    tag_with_pair = cache.name[len("_stage_cache__score__"):]
    source_lang, target_lang = tag_with_pair.rsplit("__", 1)[1].split("2", 1)
    dataset = ds.load_from_disk(str(cache))
    base_tag = tag_with_pair.rsplit("__", 1)[0]
    # Construct a minimal config sufficient for the report writer.
    config = Config(
        wav_dir=str(output_dir),
        source_lang=source_lang,
        target_lang=target_lang,
        output_dir=output_dir,
        translate={"backend": "kotoba-sdk", "label": base_tag},  # type: ignore[arg-type]
    )
    return write_summary(dataset=dataset, config=config, output_dir=output_dir)


def _dataset_name(dataset: str | Path) -> str:
    dataset_spec = str(dataset).rstrip("/")
    return dataset_spec.rsplit("/", 1)[-1]


def _infer_lang_pair(dataset: str | Path) -> tuple[str, str]:
    name = _dataset_name(dataset)
    match = _PAIR_RE.search(name)
    if not match:
        raise ValueError(
            "could not infer source/target language pair from dataset name "
            f"{name!r}; pass source_lang and target_lang explicitly"
        )
    return match.group(1), match.group(2)


def render_summary_from_dataset(
    dataset: str | Path,
    *,
    source_lang: str | None = None,
    target_lang: str | None = None,
    output_dir: str | Path | None = None,
    split: str = "train",
    label: str | None = None,
) -> dict[str, Path]:
    """Render summary files from an already-scored HF or local dataset.

    This does not run translate/transcribe/align/score. The dataset is expected
    to contain the scored columns written by the benchmark pipeline.
    """

    dataset_spec = Path(dataset).expanduser() if isinstance(dataset, Path) else dataset
    dataset_path = Path(str(dataset_spec)).expanduser()
    if dataset_path.exists() and dataset_path.is_dir():
        loaded = ds.load_from_disk(str(dataset_path))
    else:
        loaded = ds.load_dataset(str(dataset), split=split)

    if source_lang is None or target_lang is None:
        inferred_source, inferred_target = _infer_lang_pair(dataset)
        source_lang = source_lang or inferred_source
        target_lang = target_lang or inferred_target

    base_label = label or _dataset_name(dataset)
    out = Path(output_dir or (Path("./out") / base_label)).expanduser().resolve()
    config = Config(
        dataset=str(dataset),
        source_lang=source_lang,
        target_lang=target_lang,
        output_dir=out,
        translate={"backend": "kotoba-sdk", "label": base_label},  # type: ignore[arg-type]
    )
    return write_summary(dataset=loaded, config=config, output_dir=out)
