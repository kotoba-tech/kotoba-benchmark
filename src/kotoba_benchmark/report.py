"""Aggregate scored dataset into JSON / Markdown / HTML summaries."""

from __future__ import annotations

import datetime as _dt
import html
import json
import logging
import os
import statistics
import urllib.parse
from pathlib import Path
from typing import Any

import datasets as ds

from kotoba_benchmark.config import Config

logger = logging.getLogger(__name__)


_SCORE_KEYS = ("accuracy", "fluency", "conciseness")


def _classify_chunk(chunk: dict, key: str) -> tuple[float | None, str]:
    """Return (value, classification) where classification is one of 'unk', 'empty', 'ok'."""
    value = chunk.get(key) if isinstance(chunk, dict) else None
    if value is None:
        return None, "empty"
    if isinstance(value, str) and value.strip().lower() in {"unk", "n/a", ""}:
        return None, "unk"
    try:
        return float(value), "ok"
    except (TypeError, ValueError):
        return None, "unk"


def _row_score_means(row_output: list[dict] | None) -> dict[str, float | None]:
    means: dict[str, float | None] = {}
    if not row_output:
        return {f"row_{k}_mean": None for k in _SCORE_KEYS}
    for k in _SCORE_KEYS:
        vals = []
        for chunk in row_output:
            v, cls = _classify_chunk(chunk, k)
            if cls == "ok":
                vals.append(v)
        means[f"row_{k}_mean"] = statistics.fmean(vals) if vals else None
    return means


def _chunk_aggregate(
    rows_output: list[list[dict] | None],
    key: str,
    *,
    treat_empty_as_zero: bool = False,
    exclude_unk: bool = True,
) -> tuple[float | None, int]:
    values: list[float] = []
    for row in rows_output:
        if not row:
            continue
        for chunk in row:
            v, cls = _classify_chunk(chunk, key)
            if cls == "ok":
                values.append(v)
            elif cls == "empty" and treat_empty_as_zero:
                values.append(0.0)
            elif cls == "unk" and not exclude_unk:
                values.append(0.0)
    if not values:
        return None, 0
    return statistics.fmean(values), len(values)


def _collect_summary_metrics(
    dataset: ds.Dataset, config: Config
) -> dict[str, Any]:
    rows_output = dataset["output"]
    latencies = dataset["latencies"]
    translate_meta = dataset["_translate_meta"]

    rows_total = len(dataset)
    rows_with_output = sum(1 for r in rows_output if r)
    expected_chunks_total = sum(len(r) for r in rows_output if r)
    empty_chunks_total = sum(
        1 for r in rows_output if r for ch in r if _classify_chunk(ch, "accuracy")[1] == "empty"
    )
    unk_chunks_total = sum(
        1 for r in rows_output if r for ch in r if _classify_chunk(ch, "accuracy")[1] == "unk"
    )
    output_chunks_total = expected_chunks_total - empty_chunks_total

    chunk_modes = []
    for mode_key, label, kwargs in [
        ("include_unk_exclude_empty", "non-empty chunks (incl. unk as 0)",
         {"treat_empty_as_zero": False, "exclude_unk": False}),
        ("exclude_unk_exclude_empty", "non-empty, non-unk chunks",
         {"treat_empty_as_zero": False, "exclude_unk": True}),
        ("exclude_unk_empty_zero", "non-unk chunks (empty=0)",
         {"treat_empty_as_zero": True, "exclude_unk": True}),
    ]:
        mode = {"key": mode_key, "label": label, "metrics": {}}
        for k in _SCORE_KEYS:
            mean, count = _chunk_aggregate(rows_output, k, **kwargs)
            mode["metrics"][k] = {"mean": mean, "count": count}
        chunk_modes.append(mode)

    row_means_per_row = [_row_score_means(r) for r in rows_output]
    row_accuracy_mean = _agg_means([m["row_accuracy_mean"] for m in row_means_per_row])
    row_fluency_mean = _agg_means([m["row_fluency_mean"] for m in row_means_per_row])
    row_conciseness_mean = _agg_means([m["row_conciseness_mean"] for m in row_means_per_row])

    all_latencies = [v for row_lats in latencies for v in row_lats]
    median_latency = statistics.median(all_latencies) if all_latencies else None

    # translate-stage stats
    ok_count = sum(1 for m in translate_meta if m and m.get("ok"))
    first_chunk_latencies = [
        m["first_chunk_latency_s"] for m in translate_meta
        if m and m.get("first_chunk_latency_s") is not None
    ]
    median_first_chunk = (
        statistics.median(first_chunk_latencies) if first_chunk_latencies else None
    )

    return {
        "rows_total": rows_total,
        "rows_with_output": rows_with_output,
        "rows_with_output_ratio": (
            rows_with_output / rows_total if rows_total else None
        ),
        "expected_chunks_total": expected_chunks_total,
        "output_chunks_total": output_chunks_total,
        "empty_chunks_total": empty_chunks_total,
        "unk_chunks_total": unk_chunks_total,
        "empty_chunk_ratio": (
            empty_chunks_total / expected_chunks_total if expected_chunks_total else None
        ),
        "row_accuracy_mean": row_accuracy_mean,
        "row_fluency_mean": row_fluency_mean,
        "row_conciseness_mean": row_conciseness_mean,
        "median_latency_chunk": median_latency,
        "median_first_chunk_latency_translate": median_first_chunk,
        "chunk_summary_modes": chunk_modes,
        "translate_ok": ok_count,
        "translate_failed": rows_total - ok_count,
    }


def _agg_means(values: list[float | None]) -> float | None:
    filtered = [v for v in values if isinstance(v, (int, float))]
    return statistics.fmean(filtered) if filtered else None


def _audio_entries(
    dataset: ds.Dataset, config: Config, output_dir: Path
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    rows_output = dataset["output"]
    translate_meta = dataset["_translate_meta"]
    translations = dataset["translation_text"]
    ids = dataset["id"] if "id" in dataset.column_names else list(range(len(dataset)))

    for i in range(len(dataset)):
        meta = translate_meta[i] or {}
        means = _row_score_means(rows_output[i])
        entries.append({
            "id": str(ids[i]),
            "ok": meta.get("ok", False),
            "translation_text": translations[i] if i < len(translations) else "",
            "output_audio_path": meta.get("output_wav"),
            "first_chunk_latency_s": meta.get("first_chunk_latency_s"),
            "n_chunks_sts": meta.get("n_chunks"),
            "n_chunks_scored": len(rows_output[i]) if rows_output[i] else 0,
            **means,
        })
    return entries


def _format_optional(value: float | None, fmt: str = ".3f") -> str:
    if value is None:
        return "—"
    return format(value, fmt)


def _build_markdown(summary: dict[str, Any]) -> str:
    run = summary["run"]
    m = summary["metrics"]

    lines = []
    lines.append(f"# {run['tag']}\n")
    lines.append(f"- created_at: `{run['created_at']}`")
    lines.append(f"- source → target: `{run['source_lang']} → {run['target_lang']}`")
    lines.append(f"- translate backend: `{run['translate_backend']}`")
    if run.get("translate_url"):
        lines.append(f"- url: `{run['translate_url']}`")
    lines.append(f"- transcribe model: `{run['transcribe_model']}`")
    lines.append(f"- align model: `{run['align_model']}` @ `{run['align_prompt']}`")
    lines.append(f"- evaluate model: `{run['evaluate_model']}` @ `{run['evaluate_prompt']}`")
    lines.append("")

    lines.append("## Coverage\n")
    lines.append(f"- rows: {m['rows_with_output']}/{m['rows_total']} scored")
    lines.append(f"- translate stage: {m['translate_ok']} ok, {m['translate_failed']} failed")
    lines.append(
        f"- chunks: {m['output_chunks_total']}/{m['expected_chunks_total']} non-empty"
        f" ({_format_optional(m['empty_chunk_ratio'], '.1%')} empty)"
    )
    if m["unk_chunks_total"]:
        lines.append(f"- unk chunks: {m['unk_chunks_total']}")
    lines.append("")

    lines.append("## Chunk-level scores\n")
    lines.append("| mode | accuracy | fluency | conciseness | n |")
    lines.append("|---|---|---|---|---|")
    for mode in m["chunk_summary_modes"]:
        scores = mode["metrics"]
        n = scores["accuracy"]["count"]
        lines.append(
            f"| {mode['label']} | "
            f"{_format_optional(scores['accuracy']['mean'])} | "
            f"{_format_optional(scores['fluency']['mean'])} | "
            f"{_format_optional(scores['conciseness']['mean'])} | {n} |"
        )
    lines.append("")

    lines.append("## Row-level means\n")
    lines.append(f"- accuracy: {_format_optional(m['row_accuracy_mean'])}")
    lines.append(f"- fluency: {_format_optional(m['row_fluency_mean'])}")
    lines.append(f"- conciseness: {_format_optional(m['row_conciseness_mean'])}")
    lines.append("")

    lines.append("## Latency\n")
    lines.append(
        f"- median chunk start-latency (accuracy==1): "
        f"{_format_optional(m['median_latency_chunk'])} s"
    )
    lines.append(
        f"- median first-chunk latency (translate): "
        f"{_format_optional(m['median_first_chunk_latency_translate'])} s"
    )
    lines.append("")
    return "\n".join(lines)


def _build_html(summary: dict[str, Any], output_path: Path) -> str:
    run = summary["run"]
    m = summary["metrics"]
    entries = summary["audio_entries"]
    base = output_path.parent

    def _href(path: str | None) -> str:
        if not path:
            return ""
        try:
            rel = os.path.relpath(path, base)
        except ValueError:
            rel = path
        return urllib.parse.quote(rel)

    def _row(mode):
        scores = mode["metrics"]
        return (
            f"<tr><td>{html.escape(mode['label'])}</td>"
            f"<td>{_format_optional(scores['accuracy']['mean'])}</td>"
            f"<td>{_format_optional(scores['fluency']['mean'])}</td>"
            f"<td>{_format_optional(scores['conciseness']['mean'])}</td>"
            f"<td>{scores['accuracy']['count']}</td></tr>"
        )

    def _entry_row(e):
        url = _href(e["output_audio_path"])
        audio_html = (
            f'<audio controls preload="none" src="{url}"></audio>' if url else "—"
        )
        return (
            f"<tr><td>{html.escape(e['id'])}</td>"
            f"<td>{audio_html}</td>"
            f"<td>{html.escape((e['translation_text'] or '')[:120])}</td>"
            f"<td>{_format_optional(e.get('row_accuracy_mean'))}</td>"
            f"<td>{_format_optional(e.get('row_fluency_mean'))}</td>"
            f"<td>{_format_optional(e.get('row_conciseness_mean'))}</td>"
            f"<td>{_format_optional(e.get('first_chunk_latency_s'))}</td></tr>"
        )

    chunk_rows = "\n".join(_row(mode) for mode in m["chunk_summary_modes"])
    entry_rows = "\n".join(_entry_row(e) for e in entries)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(run['tag'])} — kotoba-benchmark</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #1c1e21; }}
  h1 {{ border-bottom: 1px solid #ddd; padding-bottom: .3em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #e1e4e8; padding: .5em .8em; text-align: left; vertical-align: top; }}
  th {{ background: #f6f8fa; }}
  audio {{ height: 28px; }}
  .meta {{ background: #f9f9f9; padding: 1em; border-radius: 6px; }}
  .meta code {{ background: #eee; padding: 0 .3em; border-radius: 3px; }}
</style>
</head>
<body>
<h1>{html.escape(run['tag'])}</h1>
<div class="meta">
  <div>created: <code>{html.escape(run['created_at'])}</code></div>
  <div>pair: <code>{html.escape(run['source_lang'])} → {html.escape(run['target_lang'])}</code></div>
  <div>translate backend: <code>{html.escape(run['translate_backend'])}</code></div>
  {'<div>url: <code>' + html.escape(run.get('translate_url') or '') + '</code></div>' if run.get('translate_url') else ''}
  <div>transcribe: <code>{html.escape(run['transcribe_model'])}</code></div>
  <div>align: <code>{html.escape(run['align_model'])}</code> @ <code>{html.escape(run['align_prompt'])}</code></div>
  <div>evaluate: <code>{html.escape(run['evaluate_model'])}</code> @ <code>{html.escape(run['evaluate_prompt'])}</code></div>
</div>

<h2>Coverage</h2>
<p>
  rows scored: <b>{m['rows_with_output']}</b> / {m['rows_total']} &middot;
  translate ok: <b>{m['translate_ok']}</b> ({m['translate_failed']} failed) &middot;
  non-empty chunks: <b>{m['output_chunks_total']}</b> / {m['expected_chunks_total']} (empty {_format_optional(m['empty_chunk_ratio'], '.1%')})
</p>

<h2>Chunk-level scores</h2>
<table>
<thead><tr><th>mode</th><th>accuracy</th><th>fluency</th><th>conciseness</th><th>n</th></tr></thead>
<tbody>
{chunk_rows}
</tbody>
</table>

<h2>Row-level means</h2>
<table>
<thead><tr><th>metric</th><th>value</th></tr></thead>
<tbody>
<tr><td>accuracy</td><td>{_format_optional(m['row_accuracy_mean'])}</td></tr>
<tr><td>fluency</td><td>{_format_optional(m['row_fluency_mean'])}</td></tr>
<tr><td>conciseness</td><td>{_format_optional(m['row_conciseness_mean'])}</td></tr>
<tr><td>median chunk start-latency (acc==1, s)</td><td>{_format_optional(m['median_latency_chunk'])}</td></tr>
<tr><td>median first-chunk latency translate (s)</td><td>{_format_optional(m['median_first_chunk_latency_translate'])}</td></tr>
</tbody>
</table>

<h2>Per-clip results</h2>
<table>
<thead><tr><th>id</th><th>output audio</th><th>translation</th><th>acc</th><th>flu</th><th>con</th><th>1st-chunk lat (s)</th></tr></thead>
<tbody>
{entry_rows}
</tbody>
</table>
</body>
</html>
"""


def build_summary(
    *, dataset: ds.Dataset, config: Config, output_dir: Path
) -> dict[str, Any]:
    align_prompt = find_prompt_label("align", config)
    evaluate_prompt = find_prompt_label("evaluate", config)
    return {
        "run": {
            "tag": config.dataset_tag(),
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "source_lang": config.source_lang,
            "target_lang": config.target_lang,
            "translate_backend": config.translate.backend,
            "translate_url": config.translate.url,
            "transcribe_model": config.transcribe.model,
            "align_model": config.align.model,
            "align_prompt": align_prompt,
            "evaluate_model": config.evaluate.model,
            "evaluate_prompt": evaluate_prompt,
        },
        "metrics": _collect_summary_metrics(dataset, config),
        "audio_entries": _audio_entries(dataset, config, output_dir),
    }


def find_prompt_label(kind: str, config: Config) -> str:
    from kotoba_benchmark._prompts import find_prompt
    cfg_section = config.align if kind == "align" else config.evaluate
    try:
        return find_prompt(
            kind,
            source_lang=config.source_lang,
            target_lang=config.target_lang,
            version=cfg_section.version,
        ).stem
    except FileNotFoundError:
        return "n/a"


def write_summary(
    *,
    dataset: ds.Dataset,
    config: Config,
    output_dir: Path,
) -> dict[str, Path]:
    """Write summary JSON / MD / HTML according to config.write_summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(dataset=dataset, config=config, output_dir=output_dir)
    tag = config.dataset_tag()
    written: dict[str, Path] = {}

    mode = config.write_summary
    if mode == "none":
        return written

    json_path = output_dir / f"{tag}__summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    written["json"] = json_path
    logger.info("wrote %s", json_path)

    if mode in {"json+md", "json+md+html"}:
        md_path = output_dir / f"{tag}__summary.md"
        md_path.write_text(_build_markdown(summary), encoding="utf-8")
        written["md"] = md_path
        logger.info("wrote %s", md_path)

    if mode == "json+md+html":
        html_path = output_dir / f"{tag}__summary.html"
        html_path.write_text(_build_html(summary, html_path), encoding="utf-8")
        written["html"] = html_path
        logger.info("wrote %s", html_path)

    return written
