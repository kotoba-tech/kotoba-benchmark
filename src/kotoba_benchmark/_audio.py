"""Audio helpers: turn a WAV directory into a Hugging Face Dataset."""

from __future__ import annotations

from pathlib import Path

import datasets as ds

SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3"}


def dataset_from_wav_dir(
    wav_dir: str | Path,
    *,
    source_lang: str,
    id_field: str = "id",
) -> ds.Dataset:
    """Build a Dataset from a directory tree of audio files.

    Recursively walks `wav_dir`, skipping hidden directories (anything whose
    name starts with `.`) and pipeline output caches (`_stage_cache__*`,
    `*__output_s2s`). The `id` for each row is the audio file's path relative
    to `wav_dir`, with separators replaced by `__` and the extension dropped —
    e.g. `speaker_03/utt_001.wav` → `speaker_03__utt_001`. This keeps ids
    unique under nesting and human-readable in reports.

    The returned dataset has columns:
      - `id` (str): relative path with sep -> `__`, no extension
      - `audio_<source_lang>` (Audio): decoded audio dict {"array", "sampling_rate"}
      - `source_index` (int): row index (0-based)
    """
    wav_dir = Path(wav_dir).expanduser().resolve()
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"wav_dir does not exist or is not a directory: {wav_dir}")

    def _included(p: Path) -> bool:
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
            return False
        rel_parts = p.relative_to(wav_dir).parts
        # Drop hidden dirs / files (.git, .DS_Store, etc.) and our own caches.
        for part in rel_parts:
            if part.startswith("."):
                return False
            if part.startswith("_stage_cache__") or part.endswith("__output_s2s"):
                return False
        return True

    paths = sorted(p for p in wav_dir.rglob("*") if _included(p))
    if not paths:
        raise FileNotFoundError(
            f"No audio files found under {wav_dir} (looked for {sorted(SUPPORTED_AUDIO_EXTS)})"
        )

    ids = [
        str(p.relative_to(wav_dir).with_suffix("")).replace("/", "__")
        for p in paths
    ]

    audio_column = f"audio_{source_lang}"
    rows = {
        id_field: ids,
        audio_column: [str(p) for p in paths],
        "source_index": list(range(len(paths))),
    }
    dataset = ds.Dataset.from_dict(rows)
    dataset = dataset.cast_column(audio_column, ds.Audio(decode=True))
    return dataset
