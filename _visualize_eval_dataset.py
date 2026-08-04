#!/usr/bin/env python3
"""Render report files for an already-scored evaluation dataset.

Examples:
  uv run python _visualize_eval_dataset.py kotoba-speech/example__en2ja__bench
  uv run python _visualize_eval_dataset.py ./out/run/_stage_cache__score__run__en2ja
  uv run python _visualize_eval_dataset.py ./scored_dataset --source-lang en --target-lang ja --delay 10
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys
import urllib.parse
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))


_PAIR_RE = re.compile(r"(?:^|__)([a-z]{2,3})2([a-z]{2,3})(?:__|$)")
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _dataset_name(dataset: str | Path) -> str:
    dataset_spec = str(dataset).rstrip("/")
    return dataset_spec.rsplit("/", 1)[-1]


def _infer_lang_pair(dataset: str | Path) -> tuple[str, str]:
    name = _dataset_name(dataset)
    match = _PAIR_RE.search(name)
    if not match:
        raise SystemExit(
            "Could not infer source/target language pair from dataset name "
            f"{name!r}; pass --source-lang and --target-lang."
        )
    return match.group(1), match.group(2)


def _infer_backend(label: str) -> str:
    return "openai-realtime" if "openai" in label.lower() else "kotoba-sdk"


def _load_scored_dataset(dataset: str, split: str):
    import datasets as ds

    path = Path(dataset).expanduser()
    if path.exists() and path.is_dir():
        loaded = ds.load_from_disk(str(path))
        if isinstance(loaded, ds.DatasetDict):
            if split not in loaded:
                raise SystemExit(
                    f"Local dataset has splits {sorted(loaded.keys())}, but {split!r} was requested."
                )
            return loaded[split]
        return loaded
    return ds.load_dataset(dataset, split=split)


def _timestamp_column(column_names: set[str], lang: str) -> str | None:
    for name in (f"chunked_timestamps_{lang}", f"timestamps_{lang}"):
        if name in column_names:
            return name
    return None


def _validate_columns(dataset: Any, source_lang: str, target_lang: str, backend: str, delay: float | None) -> None:
    column_names = set(dataset.column_names)
    if "output" not in column_names:
        raise SystemExit(
            "Dataset is missing required scored column 'output'. "
            "Run the benchmark through align/score first, then render this helper."
        )

    source_col = _timestamp_column(column_names, source_lang)
    target_col = _timestamp_column(column_names, target_lang)
    if not source_col or not target_col:
        missing = []
        if not source_col:
            missing.append(f"source timestamps for {source_lang}")
        if not target_col:
            missing.append(f"target timestamps for {target_lang}")
        print(
            "Warning: missing "
            + ", ".join(missing)
            + "; the HTML report will still render, but timeline bars may be incomplete.",
            file=sys.stderr,
        )

    if "_translate_meta" not in column_names and backend == "kotoba-sdk" and delay is None:
        print(
            "Warning: _translate_meta is missing and no --delay was passed; "
            "Kotoba target timestamps will not include a live timeline delay offset.",
            file=sys.stderr,
        )


def _safe_filename(value: Any, fallback: str) -> str:
    name = _SAFE_FILENAME_RE.sub("_", str(value or "").strip()).strip("._")
    if not name:
        name = fallback
    return name[:160]


def _row_ids(dataset: Any) -> list[str]:
    if "id" in dataset.column_names:
        return [str(value) if value is not None else str(index) for index, value in enumerate(dataset["id"])]
    return [str(index) for index in range(len(dataset))]


def _href(path: Path | None, base: Path) -> str:
    if path is None:
        return ""
    try:
        rel = os.path.relpath(path, base)
    except ValueError:
        rel = str(path)
    return urllib.parse.quote(rel)


def _copy_audio_file(source: str | Path, destination_stem: Path) -> Path | None:
    source_path = Path(source).expanduser()
    if not source_path.exists() or not source_path.is_file():
        return None
    suffix = source_path.suffix or ".wav"
    destination = destination_stem.with_suffix(suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination.resolve():
        shutil.copy2(source_path, destination)
    return destination


def _write_audio_value(value: Any, destination_stem: Path) -> Path | None:
    if isinstance(value, (str, Path)):
        return _copy_audio_file(value, destination_stem)

    if hasattr(value, "get_all_samples"):
        import soundfile as sf

        samples = value.get_all_samples()
        array = getattr(samples, "data", None)
        sampling_rate = getattr(samples, "sample_rate", None)
        if array is None or sampling_rate is None:
            return None
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        elif hasattr(array, "numpy"):
            array = array.numpy()
        if getattr(array, "ndim", 0) == 2 and array.shape[0] <= 8:
            array = array.T
        destination = destination_stem.with_suffix(".wav")
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, array, int(sampling_rate), subtype="PCM_16")
        return destination

    if not isinstance(value, dict):
        return None

    array = value.get("array")
    sampling_rate = value.get("sampling_rate")
    if array is not None and sampling_rate:
        import soundfile as sf

        destination = destination_stem.with_suffix(".wav")
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, array, int(sampling_rate), subtype="PCM_16")
        return destination

    path = value.get("path")
    if path:
        copied = _copy_audio_file(path, destination_stem)
        if copied is not None:
            return copied

    encoded = value.get("bytes")
    if isinstance(encoded, bytes):
        suffix = Path(str(path)).suffix if path else ".audio"
        destination = destination_stem.with_suffix(suffix or ".audio")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)
        return destination

    return None


def _export_audio_column(
    *,
    dataset: Any,
    column: str,
    row_ids: list[str],
    destination_dir: Path,
    limit: int | None,
) -> dict[int, Path]:
    if column not in dataset.column_names:
        return {}

    destination_dir.mkdir(parents=True, exist_ok=True)
    audio_view = dataset.select_columns([column])
    row_count = len(dataset) if limit is None else min(len(dataset), limit)
    written: dict[int, Path] = {}
    for index in range(row_count):
        stem = destination_dir / f"{index:05d}_{_safe_filename(row_ids[index], str(index))}"
        try:
            path = _write_audio_value(audio_view[index][column], stem)
        except Exception as exc:  # noqa: BLE001 - private inspection helper should continue.
            print(f"Warning: failed to export {column} for row {index}: {exc}", file=sys.stderr)
            continue
        if path is not None:
            written[index] = path
    return written


def _export_output_wavs_from_meta(
    *,
    dataset: Any,
    row_ids: list[str],
    destination_dir: Path,
    existing: dict[int, Path],
    limit: int | None,
) -> dict[int, Path]:
    if "_translate_meta" not in dataset.column_names:
        return {}

    destination_dir.mkdir(parents=True, exist_ok=True)
    row_count = len(dataset) if limit is None else min(len(dataset), limit)
    copied: dict[int, Path] = {}
    metas = dataset["_translate_meta"]
    for index in range(row_count):
        if index in existing:
            continue
        meta = metas[index] if index < len(metas) else None
        if not isinstance(meta, dict) or not meta.get("output_wav"):
            continue
        stem = destination_dir / f"{index:05d}_{_safe_filename(row_ids[index], str(index))}"
        copied_path = _copy_audio_file(meta["output_wav"], stem)
        if copied_path is not None:
            copied[index] = copied_path
    return copied


def _with_report_output_audio_paths(dataset: Any, target_paths: dict[int, Path]) -> Any:
    if not target_paths:
        return dataset

    if "_translate_meta" in dataset.column_names:
        metas = dataset["_translate_meta"]
    else:
        metas = [{} for _ in range(len(dataset))]

    updated = []
    for index in range(len(dataset)):
        meta = metas[index] if index < len(metas) else {}
        item = dict(meta) if isinstance(meta, dict) else {}
        if index in target_paths:
            item["output_wav"] = str(target_paths[index])
        updated.append(item)

    if "_translate_meta" in dataset.column_names:
        dataset = dataset.remove_columns(["_translate_meta"])
    return dataset.add_column("_translate_meta", updated)


def _translation_texts(dataset: Any) -> list[str]:
    if "translation_text" not in dataset.column_names:
        return ["" for _ in range(len(dataset))]
    return [str(value or "") for value in dataset["translation_text"]]


def _write_audio_check_page(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    audio_dir: Path,
    label: str,
    source_lang: str,
    target_lang: str,
) -> tuple[Path, Path]:
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = audio_dir / "manifest.json"
    page_path = output_dir / f"{label}__audio_check.html"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "label": label,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "audio_dir": str(audio_dir),
        "rows": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def _player(path: str | None) -> str:
        if not path:
            return "-"
        return f'<audio controls preload="none" src="{_href(Path(path), page_path.parent)}"></audio>'

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{row['index']}</td>"
            f"<td>{html.escape(row['id'])}</td>"
            f"<td>{_player(row.get('source_audio'))}</td>"
            f"<td>{_player(row.get('target_audio'))}</td>"
            f"<td>{html.escape((row.get('translation_text') or '')[:180])}</td>"
            "</tr>"
        )

    page_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(label)} audio check</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #1c1e21; }}
  h1 {{ border-bottom: 1px solid #ddd; padding-bottom: .3em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #e1e4e8; padding: .5em .8em; text-align: left; vertical-align: top; }}
  th {{ background: #f6f8fa; }}
  audio {{ height: 28px; width: 260px; }}
  code {{ background: #eee; padding: 0 .3em; border-radius: 3px; }}
</style>
</head>
<body>
<h1>{html.escape(label)} audio check</h1>
<p>Audio files were saved under <code>{html.escape(str(audio_dir))}</code>.</p>
<table>
<thead>
<tr><th>#</th><th>id</th><th>source ({html.escape(source_lang)})</th><th>target ({html.escape(target_lang)})</th><th>translation</th></tr>
</thead>
<tbody>
{''.join(table_rows)}
</tbody>
</table>
</body>
</html>
""",
        encoding="utf-8",
    )
    return manifest_path, page_path


def _save_audio_check_artifacts(
    *,
    dataset: Any,
    source_lang: str,
    target_lang: str,
    output_dir: Path,
    audio_dir: Path,
    label: str,
    limit: int | None,
) -> tuple[Any, dict[str, Path]]:
    row_ids = _row_ids(dataset)
    source_col = f"audio_{source_lang}"
    target_col = f"audio_{target_lang}"
    source_paths = _export_audio_column(
        dataset=dataset,
        column=source_col,
        row_ids=row_ids,
        destination_dir=audio_dir / "source",
        limit=limit,
    )
    target_paths = _export_audio_column(
        dataset=dataset,
        column=target_col,
        row_ids=row_ids,
        destination_dir=audio_dir / "target",
        limit=limit,
    )
    target_paths.update(
        _export_output_wavs_from_meta(
            dataset=dataset,
            row_ids=row_ids,
            destination_dir=audio_dir / "target",
            existing=target_paths,
            limit=limit,
        )
    )

    dataset = _with_report_output_audio_paths(dataset, target_paths)
    texts = _translation_texts(dataset)
    row_count = len(dataset) if limit is None else min(len(dataset), limit)
    rows = [
        {
            "index": index,
            "id": row_ids[index],
            "source_audio": str(source_paths[index]) if index in source_paths else None,
            "target_audio": str(target_paths[index]) if index in target_paths else None,
            "translation_text": texts[index] if index < len(texts) else "",
        }
        for index in range(row_count)
    ]
    manifest_path, page_path = _write_audio_check_page(
        rows=rows,
        output_dir=output_dir,
        audio_dir=audio_dir,
        label=label,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    return dataset, {"audio_manifest": manifest_path, "audio_html": page_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render JSON/Markdown/HTML reports from an already-scored evaluation dataset."
    )
    parser.add_argument(
        "dataset",
        help="Hugging Face dataset repo ID or local load_from_disk dataset directory.",
    )
    parser.add_argument("--split", default="train", help="Split to load for Hugging Face or DatasetDict inputs.")
    parser.add_argument("--source-lang", default=None, help="Source language code, such as en.")
    parser.add_argument("--target-lang", default=None, help="Target language code, such as ja.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated summary files.")
    parser.add_argument("--label", default=None, help="Label used in output filenames.")
    parser.add_argument(
        "--backend",
        default=None,
        help="Translation backend for report metadata and timeline offsets, e.g. kotoba-sdk or openai-realtime.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Kotoba delay setting. The report converts this to delay * 0.08 seconds on the target timeline.",
    )
    parser.add_argument(
        "--summary-mode",
        choices=("json", "json+md", "json+md+html"),
        default="json+md+html",
        help="Which summary files to write.",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Export source/target audio to WAV files and write an audio check HTML page.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Directory for exported audio. Passing this also enables --save-audio.",
    )
    parser.add_argument(
        "--audio-limit",
        type=int,
        default=None,
        help="Export audio for at most this many rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_lang = args.source_lang
    target_lang = args.target_lang
    if source_lang is None or target_lang is None:
        inferred_source, inferred_target = _infer_lang_pair(args.dataset)
        source_lang = source_lang or inferred_source
        target_lang = target_lang or inferred_target

    label = args.label or _dataset_name(args.dataset)
    backend = args.backend or _infer_backend(label)
    output_dir = Path(args.output_dir or (Path("out") / label)).expanduser().resolve()

    dataset = _load_scored_dataset(args.dataset, args.split)
    _validate_columns(dataset, source_lang, target_lang, backend, args.delay)
    extra_paths: dict[str, Path] = {}
    if args.save_audio or args.audio_dir is not None:
        audio_dir = Path(args.audio_dir or (output_dir / "audio_check")).expanduser().resolve()
        dataset, extra_paths = _save_audio_check_artifacts(
            dataset=dataset,
            source_lang=source_lang,
            target_lang=target_lang,
            output_dir=output_dir,
            audio_dir=audio_dir,
            label=label,
            limit=args.audio_limit,
        )

    from kotoba_benchmark.config import Config
    from kotoba_benchmark.report import write_summary

    translate: dict[str, Any] = {"backend": backend, "label": label}
    if args.delay is not None:
        translate["delay"] = args.delay

    config = Config(
        dataset=args.dataset,
        source_lang=source_lang,
        target_lang=target_lang,
        output_dir=output_dir,
        translate=translate,
        write_summary=args.summary_mode,
    )
    paths = write_summary(dataset=dataset, config=config, output_dir=output_dir)
    paths.update(extra_paths)
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
    if "html" in paths:
        print(f"\nHTML: {paths['html']}")
    if "audio_html" in paths:
        print(f"Audio check: {paths['audio_html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
