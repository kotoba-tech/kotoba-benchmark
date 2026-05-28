"""Per-sample translate runner: stream every row through the backend with bounded concurrency."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import datasets as ds
import numpy as np
import soundfile as sf
from tqdm import tqdm

from kotoba_benchmark.config import Config
from kotoba_benchmark.stages.translate import TranslateBackend, get_backend

logger = logging.getLogger(__name__)


def _resample_to(audio: np.ndarray, src_rate: int, tgt_rate: int) -> np.ndarray:
    if src_rate == tgt_rate:
        return audio
    import librosa

    resampled = librosa.resample(
        audio.astype(np.float32) / 32768.0, orig_sr=src_rate, target_sr=tgt_rate
    )
    return (np.clip(resampled, -1.0, 1.0) * 32768.0).astype(np.int16)


def _row_pcm16(audio_dict: dict, target_rate: int) -> bytes:
    """Decode an HF Audio column entry to PCM16 bytes at `target_rate`."""
    array = audio_dict["array"]
    src_rate = audio_dict["sampling_rate"]
    if array.dtype.kind == "f":
        pcm = np.clip(array, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
    elif array.dtype == np.int16:
        pcm = array
    else:
        pcm = array.astype(np.int16)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=tuple(range(pcm.ndim))[1:]).astype(np.int16)
    pcm = _resample_to(pcm, src_rate, target_rate)
    return pcm.tobytes()


async def _translate_one(
    *,
    backend: TranslateBackend,
    sample_id: str,
    pcm16: bytes,
    sample_rate: int,
    source_lang: str,
    target_lang: str,
    max_retries: int,
    retry_interval_seconds: float,
    output_audio_dir: Path,
) -> dict[str, Any]:
    """Run one translate request with retries; persist output audio + transcript."""

    output_wav_path = output_audio_dir / f"{sample_id}_output.wav"
    output_txt_path = output_audio_dir / f"{sample_id}_translation.txt"

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            audio_buffer = bytearray()
            partial_texts: list[str] = []
            chunk_arrivals: list[float] = []
            sent_at = time.monotonic()

            async for chunk in backend.translate(
                pcm16=pcm16,
                sample_rate=sample_rate,
                source_lang=source_lang,
                target_lang=target_lang,
            ):
                if chunk.audio:
                    audio_buffer.extend(chunk.audio)
                if chunk.partial_source:
                    partial_texts.append(chunk.partial_source)
                chunk_arrivals.append(chunk.received_at_monotonic - sent_at)

            audio_array = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
            sf.write(output_wav_path, audio_array, sample_rate, subtype="PCM_16")
            transcript = "".join(partial_texts).strip()
            output_txt_path.write_text(transcript, encoding="utf-8")

            return {
                "id": sample_id,
                "ok": True,
                "attempts": attempt,
                "output_wav": str(output_wav_path),
                "translation_text": transcript,
                "first_chunk_latency_s": chunk_arrivals[0] if chunk_arrivals else None,
                "last_chunk_latency_s": chunk_arrivals[-1] if chunk_arrivals else None,
                "n_chunks": len(chunk_arrivals),
            }
        except Exception as exc:  # noqa: BLE001 — backend may raise anything
            last_err = exc
            logger.warning(
                "translate attempt %d/%d failed for %s: %s",
                attempt, max_retries, sample_id, exc,
            )
            if attempt < max_retries:
                await asyncio.sleep(retry_interval_seconds)

    return {
        "id": sample_id,
        "ok": False,
        "attempts": max_retries,
        "error": repr(last_err) if last_err else "unknown",
        "output_wav": None,
        "translation_text": "",
        "first_chunk_latency_s": None,
        "last_chunk_latency_s": None,
        "n_chunks": 0,
    }


async def translate_dataset_async(
    *,
    dataset: ds.Dataset,
    config: Config,
    output_dir: Path,
) -> ds.Dataset:
    """Run the translate stage over every row, returning a new dataset with target audio + transcript.

    Output files live under `output_dir/{tag}__output_s2s/`. The returned dataset
    gets new columns: `audio_<target_lang>`, `translation_text`, `_translate_meta`.
    """

    backend_cls = get_backend(config.translate.backend)
    backend_kwargs = config.translate.model_dump(
        exclude={"backend", "max_concurrency", "max_retries", "retry_interval_seconds", "label"},
        exclude_none=True,
    )
    backend_kwargs.pop("chunk_ms", None)
    backend: TranslateBackend = backend_cls(
        url=config.translate.url,
        chunk_ms=config.translate.chunk_ms,
        **{k: v for k, v in backend_kwargs.items() if k not in {"url", "sample_rate"}},
    )

    tag = config.dataset_tag()
    output_audio_dir = output_dir / f"{tag}__output_s2s"
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    source_col = f"audio_{config.source_lang}"
    if source_col not in dataset.column_names:
        raise KeyError(f"dataset missing source audio column {source_col!r}")

    sample_rate = config.translate.sample_rate
    semaphore = asyncio.Semaphore(config.translate.max_concurrency)

    rows = [dataset[i] for i in range(len(dataset))]
    pbar = tqdm(
        total=len(rows),
        desc="translate",
        unit="file",
        disable=not config.show_progress(),
    )

    async def _process(idx: int, row: dict) -> dict[str, Any]:
        async with semaphore:
            pcm16 = _row_pcm16(row[source_col], sample_rate)
            result = await _translate_one(
                backend=backend,
                sample_id=str(row.get("id", idx)),
                pcm16=pcm16,
                sample_rate=sample_rate,
                source_lang=config.source_lang,
                target_lang=config.target_lang,
                max_retries=config.translate.max_retries,
                retry_interval_seconds=config.translate.retry_interval_seconds,
                output_audio_dir=output_audio_dir,
            )
            pbar.update(1)
            return result

    tasks = [_process(i, row) for i, row in enumerate(rows)]
    try:
        results = await asyncio.gather(*tasks)
    finally:
        pbar.close()

    ok = sum(1 for r in results if r["ok"])
    logger.info("translate stage complete: %d/%d ok", ok, len(results))

    target_col = f"audio_{config.target_lang}"
    output_wavs = [r["output_wav"] for r in results]
    translations = [r["translation_text"] for r in results]
    meta = [
        {
            k: r[k]
            for k in ("attempts", "first_chunk_latency_s", "last_chunk_latency_s", "n_chunks", "ok", "output_wav")
        }
        for r in results
    ]

    new_dataset = dataset.add_column(target_col, output_wavs)
    new_dataset = new_dataset.cast_column(target_col, ds.Audio(decode=True))
    new_dataset = new_dataset.add_column("translation_text", translations)
    new_dataset = new_dataset.add_column("_translate_meta", meta)

    (output_audio_dir / "translate_metadata.json").write_text(
        json.dumps(
            {
                "tag": tag,
                "source_lang": config.source_lang,
                "target_lang": config.target_lang,
                "backend": config.translate.backend,
                "url": config.translate.url,
                "max_concurrency": config.translate.max_concurrency,
                "max_retries": config.translate.max_retries,
                "sample_rate": sample_rate,
                "total": len(results),
                "ok": ok,
                "failed": len(results) - ok,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return new_dataset


def translate_dataset(
    *, dataset: ds.Dataset, config: Config, output_dir: Path
) -> ds.Dataset:
    """Synchronous wrapper for the translate stage."""
    return asyncio.run(
        translate_dataset_async(dataset=dataset, config=config, output_dir=output_dir)
    )
