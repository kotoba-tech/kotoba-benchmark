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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    translate_meta = (
        dataset["_translate_meta"]
        if "_translate_meta" in dataset.column_names
        else [{} for _ in range(len(dataset))]
    )
    if "latencies" in dataset.column_names:
        latencies = dataset["latencies"]
    else:
        source_col = _timestamp_column(dataset, config.source_lang)
        target_col = _timestamp_column(dataset, config.target_lang)
        source_all = dataset[source_col] if source_col else []
        target_all = dataset[target_col] if target_col else []
        latencies = [
            [
                chunk["start_latency_s"]
                for chunk in _timeline_chunks(
                    rows_output[i],
                    source_all[i] if i < len(source_all) else None,
                    target_all[i] if i < len(target_all) else None,
                )
                if chunk.get("contributes_latency")
                and chunk.get("start_latency_s") is not None
            ]
            for i in range(len(dataset))
        ]

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


def _timestamp_column(dataset: ds.Dataset, lang: str) -> str | None:
    chunked_col = f"chunked_timestamps_{lang}"
    if chunked_col in dataset.column_names:
        return chunked_col
    timestamp_col = f"timestamps_{lang}"
    if timestamp_col in dataset.column_names:
        return timestamp_col
    return None


def _audio_entries(
    dataset: ds.Dataset, config: Config, output_dir: Path
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    rows_output = dataset["output"]
    translate_meta = (
        dataset["_translate_meta"]
        if "_translate_meta" in dataset.column_names
        else [{} for _ in range(len(dataset))]
    )
    translations = (
        dataset["translation_text"]
        if "translation_text" in dataset.column_names
        else ["" for _ in range(len(dataset))]
    )
    ids = dataset["id"] if "id" in dataset.column_names else list(range(len(dataset)))
    source_chunks_col = _timestamp_column(dataset, config.source_lang)
    target_chunks_col = _timestamp_column(dataset, config.target_lang)
    source_chunks_all = dataset[source_chunks_col] if source_chunks_col else []
    target_chunks_all = dataset[target_chunks_col] if target_chunks_col else []

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
            "timeline_chunks": _timeline_chunks(
                rows_output[i],
                source_chunks_all[i] if i < len(source_chunks_all) else None,
                target_chunks_all[i] if i < len(target_chunks_all) else None,
            ),
            **means,
        })
    return entries


def _timeline_chunks(
    outputs: list[dict] | None,
    source_chunks: list[dict] | None,
    target_chunks: list[dict] | None,
) -> list[dict[str, Any]]:
    """Build per-chunk timeline records and mirror the latency inclusion rule."""
    if not isinstance(outputs, list):
        outputs = []
    if not isinstance(source_chunks, list):
        source_chunks = []
    if not isinstance(target_chunks, list):
        target_chunks = []

    chunks: list[dict[str, Any]] = []
    for idx in range(max(len(outputs), len(source_chunks), len(target_chunks))):
        out = outputs[idx] if idx < len(outputs) and isinstance(outputs[idx], dict) else {}
        src = (
            source_chunks[idx]
            if idx < len(source_chunks) and isinstance(source_chunks[idx], dict)
            else {}
        )
        tgt = (
            target_chunks[idx]
            if idx < len(target_chunks) and isinstance(target_chunks[idx], dict)
            else {}
        )
        src_start = src.get("start")
        src_end = src.get("end")
        tgt_start = tgt.get("start")
        tgt_end = tgt.get("end")
        accuracy = out.get("accuracy")
        valid_timestamps = all(
            _is_number(v) for v in (src_start, src_end, tgt_start, tgt_end)
        )
        start_latency = (
            float(tgt_start) - float(src_start) if valid_timestamps else None
        )
        end_latency = float(tgt_end) - float(src_end) if valid_timestamps else None
        contributes_latency = (
            _is_number(accuracy)
            and float(accuracy) == 1.0
            and start_latency is not None
            and end_latency is not None
            and start_latency >= 0
            and end_latency >= 0
        )
        chunks.append(
            {
                "index": idx,
                "source": {
                    "text": str(src.get("text") or ""),
                    "start": float(src_start) if _is_number(src_start) else None,
                    "end": float(src_end) if _is_number(src_end) else None,
                },
                "target": {
                    "text": str(tgt.get("text") or ""),
                    "start": float(tgt_start) if _is_number(tgt_start) else None,
                    "end": float(tgt_end) if _is_number(tgt_end) else None,
                },
                "accuracy": float(accuracy) if _is_number(accuracy) else None,
                "start_latency_s": start_latency if contributes_latency else None,
                "end_latency_s": end_latency if contributes_latency else None,
                "contributes_latency": contributes_latency,
            }
        )
    return chunks


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

    timeline_left = 54.0
    timeline_right = 40.0
    timeline_min_width = 980.0
    timeline_px_per_second = 120.0
    timeline_px_per_chunk = 170.0
    timeline_height = 168
    bar_height = 52
    source_y = 18
    target_y = 92
    source_center_y = source_y + bar_height / 2
    target_center_y = target_y + bar_height / 2
    overview_width = 980.0
    overview_height = 96
    overview_left = 40.0
    overview_right = 40.0
    overview_bar_height = 16
    overview_source_y = 19
    overview_target_y = 60
    overview_source_center_y = overview_source_y + overview_bar_height / 2
    overview_target_center_y = overview_target_y + overview_bar_height / 2

    def _scale_range(
        value: float,
        min_time: float,
        span: float,
        width: float,
        left: float,
        right: float,
    ) -> float:
        drawable_width = width - left - right
        return left + ((value - min_time) / span) * drawable_width

    def _scale(
        value: float, min_time: float, span: float, timeline_width: float
    ) -> float:
        return _scale_range(
            value, min_time, span, timeline_width, timeline_left, timeline_right
        )

    def _overview_scale(value: float, min_time: float, span: float) -> float:
        return _scale_range(
            value, min_time, span, overview_width, overview_left, overview_right
        )

    def _bar(
        *,
        chunk: dict[str, Any],
        side: str,
        min_time: float,
        span: float,
        timeline_width: float,
        y: int,
        color: str,
    ) -> str:
        data = chunk.get(side) if isinstance(chunk, dict) else {}
        if not isinstance(data, dict):
            return ""
        start = data.get("start")
        end = data.get("end")
        if not _is_number(start) or not _is_number(end):
            return ""
        x = _scale(float(start), min_time, span, timeline_width)
        w = max(_scale(float(end), min_time, span, timeline_width) - x, 8.0)
        opacity = "0.86" if chunk.get("contributes_latency") else "0.34"
        label = str(data.get("text") or "")
        title = (
            f"{side} chunk {chunk.get('index', '')}: "
            f"{float(start):.3f}s-{float(end):.3f}s "
            f"{label}"
        )
        text_x = x + 6.0
        text_y = float(y) + 5.0
        text_w = max(w - 12.0, 1.0)
        text_h = bar_height - 10.0
        return (
            f"<g><title>{html.escape(title)}</title>"
            f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{bar_height}" rx="5" '
            f'fill="{color}" opacity="{opacity}"></rect>'
            f'<foreignObject x="{text_x:.2f}" y="{text_y:.2f}" width="{text_w:.2f}" '
            f'height="{text_h:.2f}" pointer-events="none">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" class="timeline-chunk-text">'
            f"{html.escape(label)}</div></foreignObject></g>"
        )

    def _overview_bar(
        *,
        chunk: dict[str, Any],
        side: str,
        min_time: float,
        span: float,
        y: int,
        color: str,
    ) -> str:
        data = chunk.get(side) if isinstance(chunk, dict) else {}
        if not isinstance(data, dict):
            return ""
        start = data.get("start")
        end = data.get("end")
        if not _is_number(start) or not _is_number(end):
            return ""
        x = _overview_scale(float(start), min_time, span)
        w = max(_overview_scale(float(end), min_time, span) - x, 2.0)
        opacity = "0.86" if chunk.get("contributes_latency") else "0.34"
        title = (
            f"{side} chunk {chunk.get('index', '')}: "
            f"{float(start):.3f}s-{float(end):.3f}s "
            f"{data.get('text') or ''}"
        )
        return (
            f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{overview_bar_height}" rx="3" '
            f'fill="{color}" opacity="{opacity}">'
            f"<title>{html.escape(title)}</title></rect>"
        )

    def _latency_marker(
        chunk: dict[str, Any], min_time: float, span: float, timeline_width: float
    ) -> str:
        if not chunk.get("contributes_latency"):
            return ""
        source = chunk.get("source") if isinstance(chunk.get("source"), dict) else {}
        target = chunk.get("target") if isinstance(chunk.get("target"), dict) else {}
        src_start = source.get("start")
        tgt_start = target.get("start")
        latency = chunk.get("start_latency_s")
        if not _is_number(src_start) or not _is_number(tgt_start) or latency is None:
            return ""
        x1 = _scale(float(src_start), min_time, span, timeline_width)
        x2 = _scale(float(tgt_start), min_time, span, timeline_width)
        width = abs(x2 - x1)
        mid = min(x1, x2)
        title = f"calculated latency for chunk {chunk.get('index', '')}: {latency:.3f}s"
        return (
            f'<rect x="{mid:.2f}" y="75" width="{max(width, 2.0):.2f}" height="10" rx="5" '
            f'fill="#d62728" opacity="0.38"><title>{html.escape(title)}</title></rect>'
            f'<line x1="{x1:.2f}" y1="{source_center_y:.2f}" x2="{x2:.2f}" y2="{target_center_y:.2f}" '
            f'stroke="#d62728" stroke-width="2.2">'
            f"<title>{html.escape(title)}</title></line>"
            f'<circle cx="{x1:.2f}" cy="{source_center_y:.2f}" r="3.5" fill="#d62728"><title>{html.escape(title)}</title></circle>'
            f'<circle cx="{x2:.2f}" cy="{target_center_y:.2f}" r="3.5" fill="#d62728"><title>{html.escape(title)}</title></circle>'
        )

    def _latency_label(
        chunk: dict[str, Any], min_time: float, span: float, timeline_width: float
    ) -> str:
        if not chunk.get("contributes_latency"):
            return ""
        source = chunk.get("source") if isinstance(chunk.get("source"), dict) else {}
        target = chunk.get("target") if isinstance(chunk.get("target"), dict) else {}
        src_start = source.get("start")
        tgt_start = target.get("start")
        latency = chunk.get("start_latency_s")
        if not _is_number(src_start) or not _is_number(tgt_start) or latency is None:
            return ""
        x1 = _scale(float(src_start), min_time, span, timeline_width)
        x2 = _scale(float(tgt_start), min_time, span, timeline_width)
        width = abs(x2 - x1)
        if width < 46.0:
            return ""
        mid = min(x1, x2)
        title = f"calculated latency for chunk {chunk.get('index', '')}: {latency:.3f}s"
        return (
            f'<text x="{mid + width / 2:.2f}" y="83" class="timeline-latency-text">'
            f"<title>{html.escape(title)}</title>{latency:.3f}s</text>"
        )

    def _overview_latency_marker(
        chunk: dict[str, Any], min_time: float, span: float
    ) -> str:
        if not chunk.get("contributes_latency"):
            return ""
        source = chunk.get("source") if isinstance(chunk.get("source"), dict) else {}
        target = chunk.get("target") if isinstance(chunk.get("target"), dict) else {}
        src_start = source.get("start")
        tgt_start = target.get("start")
        latency = chunk.get("start_latency_s")
        if not _is_number(src_start) or not _is_number(tgt_start) or latency is None:
            return ""
        x1 = _overview_scale(float(src_start), min_time, span)
        x2 = _overview_scale(float(tgt_start), min_time, span)
        width = abs(x2 - x1)
        mid = min(x1, x2)
        title = f"calculated latency for chunk {chunk.get('index', '')}: {latency:.3f}s"
        return (
            f'<rect x="{mid:.2f}" y="42" width="{max(width, 2.0):.2f}" height="8" rx="4" '
            f'fill="#d62728" opacity="0.28"><title>{html.escape(title)}</title></rect>'
            f'<line x1="{x1:.2f}" y1="{overview_source_center_y:.2f}" x2="{x2:.2f}" y2="{overview_target_center_y:.2f}" '
            f'stroke="#d62728" stroke-width="2.2">'
            f"<title>{html.escape(title)}</title></line>"
            f'<circle cx="{x1:.2f}" cy="{overview_source_center_y:.2f}" r="3.5" fill="#d62728"><title>{html.escape(title)}</title></circle>'
            f'<circle cx="{x2:.2f}" cy="{overview_target_center_y:.2f}" r="3.5" fill="#d62728"><title>{html.escape(title)}</title></circle>'
        )

    def _timeline(e: dict[str, Any]) -> str:
        chunks = e.get("timeline_chunks")
        if not isinstance(chunks, list) or not chunks:
            return '<div class="timeline-empty">No aligned chunk timestamps.</div>'
        times: list[float] = []
        for chunk in chunks:
            for side in ("source", "target"):
                data = chunk.get(side) if isinstance(chunk, dict) else None
                if not isinstance(data, dict):
                    continue
                for key in ("start", "end"):
                    value = data.get(key)
                    if _is_number(value):
                        times.append(float(value))
        if not times:
            return '<div class="timeline-empty">No aligned chunk timestamps.</div>'
        min_time = min(0.0, min(times))
        max_time = max(times)
        span = max(max_time - min_time, 0.001)
        timeline_width = int(
            max(
                timeline_min_width,
                span * timeline_px_per_second + timeline_left + timeline_right,
                len(chunks) * timeline_px_per_chunk + timeline_left + timeline_right,
            )
        )
        axis_end = timeline_width - timeline_right
        right_tick_x = max(timeline_left, axis_end - 42)
        markers = "\n".join(
            _latency_marker(chunk, min_time, span, timeline_width) for chunk in chunks
        )
        marker_labels = "\n".join(
            _latency_label(chunk, min_time, span, timeline_width) for chunk in chunks
        )
        source_bars = "\n".join(
            _bar(
                chunk=chunk,
                side="source",
                min_time=min_time,
                span=span,
                timeline_width=timeline_width,
                y=source_y,
                color="#4c78a8",
            )
            for chunk in chunks
        )
        target_bars = "\n".join(
            _bar(
                chunk=chunk,
                side="target",
                min_time=min_time,
                span=span,
                timeline_width=timeline_width,
                y=target_y,
                color="#f58518",
            )
            for chunk in chunks
        )
        overview_markers = "\n".join(
            _overview_latency_marker(chunk, min_time, span) for chunk in chunks
        )
        overview_source_bars = "\n".join(
            _overview_bar(
                chunk=chunk,
                side="source",
                min_time=min_time,
                span=span,
                y=overview_source_y,
                color="#4c78a8",
            )
            for chunk in chunks
        )
        overview_target_bars = "\n".join(
            _overview_bar(
                chunk=chunk,
                side="target",
                min_time=min_time,
                span=span,
                y=overview_target_y,
                color="#f58518",
            )
            for chunk in chunks
        )
        included = [
            chunk for chunk in chunks if isinstance(chunk, dict) and chunk.get("contributes_latency")
        ]
        latency_labels = ", ".join(
            f"#{chunk['index']}={chunk['start_latency_s']:.3f}s"
            for chunk in included[:8]
            if chunk.get("start_latency_s") is not None
        )
        if len(included) > 8:
            latency_labels += f", +{len(included) - 8} more"
        if not latency_labels:
            latency_labels = "no chunks contributed to median_latency_chunk"
        return f"""
<div class="timeline-wrap">
  <div class="timeline-scroll">
  <svg class="timeline-svg" width="{timeline_width}" height="{timeline_height}" viewBox="0 0 {timeline_width} {timeline_height}" role="img" aria-label="source and target chunk timeline">
    <line x1="{timeline_left:.0f}" y1="{source_center_y:.0f}" x2="{axis_end:.0f}" y2="{source_center_y:.0f}" stroke="#d8dee4" />
    <line x1="{timeline_left:.0f}" y1="{target_center_y:.0f}" x2="{axis_end:.0f}" y2="{target_center_y:.0f}" stroke="#d8dee4" />
    <text x="0" y="{source_center_y + 5:.0f}" class="timeline-label">src</text>
    <text x="0" y="{target_center_y + 5:.0f}" class="timeline-label">tgt</text>
    <text x="{timeline_left:.0f}" y="{timeline_height - 12}" class="timeline-tick">{min_time:.1f}s</text>
    <text x="{right_tick_x:.0f}" y="{timeline_height - 12}" class="timeline-tick">{max_time:.1f}s</text>
    {markers}
    {source_bars}
    {target_bars}
    {marker_labels}
  </svg>
  </div>
  <div class="timeline-overview">
    <div class="timeline-overview-label">compact overview</div>
    <svg class="timeline-overview-svg" viewBox="0 0 {overview_width:.0f} {overview_height}" role="img" aria-label="compact source and target chunk timeline">
      <line x1="{overview_left:.0f}" y1="{overview_source_center_y:.0f}" x2="{overview_width - overview_right:.0f}" y2="{overview_source_center_y:.0f}" stroke="#d8dee4" />
      <line x1="{overview_left:.0f}" y1="{overview_target_center_y:.0f}" x2="{overview_width - overview_right:.0f}" y2="{overview_target_center_y:.0f}" stroke="#d8dee4" />
      <text x="0" y="31" class="timeline-label">src</text>
      <text x="0" y="72" class="timeline-label">tgt</text>
      <text x="{overview_left:.0f}" y="93" class="timeline-tick">{min_time:.1f}s</text>
      <text x="{overview_width - overview_right - 40:.0f}" y="93" class="timeline-tick">{max_time:.1f}s</text>
      {overview_source_bars}
      {overview_target_bars}
      {overview_markers}
    </svg>
  </div>
  <div class="timeline-legend">
    <span><i class="swatch source"></i>source chunks</span>
    <span><i class="swatch target"></i>target chunks</span>
    <span><i class="swatch latency"></i>latency for successful aligned pairs</span>
    <span class="timeline-values">{html.escape(latency_labels)}</span>
  </div>
</div>
"""

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
            f"<td>{_format_optional(e.get('first_chunk_latency_s'))}</td>"
            f"<td>{_timeline(e)}</td></tr>"
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
  .timeline-wrap {{ min-width: 520px; max-width: 760px; }}
  .timeline-scroll {{ overflow-x: auto; overflow-y: hidden; max-width: 100%; border: 1px solid #e1e4e8; border-radius: 6px; background: #fff; padding: .25em; }}
  .timeline-svg {{ display: block; max-width: none; height: auto; }}
  .timeline-chunk-text {{ box-sizing: border-box; width: 100%; height: 100%; overflow: hidden; color: #fff; font-size: 12px; line-height: 1.25; font-weight: 600; text-shadow: 0 1px 1px rgba(0, 0, 0, .28); display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 3; }}
  .timeline-overview {{ margin-top: .45em; }}
  .timeline-overview-label {{ color: #57606a; font-size: 11px; font-weight: 600; margin: 0 0 .2em; }}
  .timeline-overview-svg {{ width: 100%; max-width: 980px; height: auto; display: block; border: 1px solid #e1e4e8; border-radius: 6px; background: #fff; }}
  .timeline-label {{ font-size: 13px; fill: #57606a; font-weight: 600; }}
  .timeline-tick {{ font-size: 11px; fill: #6e7781; }}
  .timeline-latency-text {{ fill: #7a1712; font-size: 11px; font-weight: 700; text-anchor: middle; dominant-baseline: middle; paint-order: stroke; stroke: #fff; stroke-width: 3px; stroke-linejoin: round; pointer-events: none; }}
  .timeline-legend {{ display: flex; flex-wrap: wrap; gap: .45em .9em; align-items: center; color: #57606a; font-size: 12px; }}
  .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: .3em; vertical-align: -1px; }}
  .swatch.source {{ background: #4c78a8; }}
  .swatch.target {{ background: #f58518; }}
  .swatch.latency {{ background: #d62728; }}
  .timeline-values {{ color: #24292f; }}
  .timeline-empty {{ color: #6e7781; font-size: 12px; }}
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
<thead><tr><th>id</th><th>output audio</th><th>translation</th><th>acc</th><th>flu</th><th>con</th><th>1st-chunk lat (s)</th><th>latency timeline</th></tr></thead>
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
