"""Command-line interface for kotoba-benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from kotoba_benchmark.config import Config


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _coerce(value: str) -> Any:
    """Best-effort coerce a CLI string value to int/float/bool/None/string."""
    s = value.strip()
    lower = s.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `--override section.field=value` strings to a config dict."""
    for ov in overrides:
        if "=" not in ov:
            raise SystemExit(f"--override must be key=value, got: {ov}")
        key, _, raw_value = ov.partition("=")
        path = key.strip().split(".")
        target = data
        for part in path[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise SystemExit(f"--override path {key!r} collides with a non-dict value")
        target[path[-1]] = _coerce(raw_value)
    return data


def _cmd_run(args: argparse.Namespace) -> int:
    import sys

    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover
        import tomli as tomllib

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise SystemExit(f"config file not found: {config_path}")

    with open(config_path, "rb") as f:
        data: dict[str, Any] = tomllib.load(f)

    if args.output_dir:
        data["output_dir"] = args.output_dir
    if args.wav_dir:
        data["wav_dir"] = args.wav_dir

    data = _apply_overrides(data, args.override or [])
    config = Config.from_dict(data)

    from kotoba_benchmark.pipeline import evaluate

    result = evaluate(config)

    print()
    print(f"Tag: {result.tag}")
    print(f"Output dir: {result.output_dir}")
    print(f"Summary: {result.summary_paths}")
    print()
    print("Scores:")
    metrics = result.summary["metrics"]
    print(f"  row_accuracy_mean: {metrics['row_accuracy_mean']}")
    print(f"  row_fluency_mean: {metrics['row_fluency_mean']}")
    print(f"  row_conciseness_mean: {metrics['row_conciseness_mean']}")
    print(f"  median_latency_chunk_s: {metrics['median_latency_chunk']}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from kotoba_benchmark.pipeline import re_render_summary

    paths = re_render_summary(args.output_dir)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


def _cmd_show_prompts(args: argparse.Namespace) -> int:
    from kotoba_benchmark._prompts import find_prompt

    pair_args = {"source_lang": args.source_lang, "target_lang": args.target_lang}
    align = find_prompt("align", **pair_args, version=args.version)
    evaluate = find_prompt("evaluate", **pair_args, version=args.version)
    print(json.dumps({"align": str(align), "evaluate": str(evaluate)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kotoba-benchmark",
        description="Speech-to-speech translation benchmark.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run an evaluation from a TOML config")
    p_run.add_argument("config", help="Path to a TOML config")
    p_run.add_argument("--output-dir", help="Override output_dir from config")
    p_run.add_argument("--wav-dir", help="Override wav_dir from config")
    p_run.add_argument(
        "--override",
        action="append",
        metavar="KEY=VALUE",
        help="Override config field (e.g. translate.url=wss://...). Repeatable.",
    )
    p_run.set_defaults(func=_cmd_run)

    p_report = sub.add_parser(
        "report", help="Re-render summary files from a previous run's output_dir"
    )
    p_report.add_argument("output_dir")
    p_report.set_defaults(func=_cmd_report)

    p_prompts = sub.add_parser(
        "show-prompts",
        help="Print the prompt TOML paths kotoba-benchmark would use for a language pair",
    )
    p_prompts.add_argument("--source-lang", required=True)
    p_prompts.add_argument("--target-lang", required=True)
    p_prompts.add_argument("--version", default=None)
    p_prompts.set_defaults(func=_cmd_show_prompts)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
