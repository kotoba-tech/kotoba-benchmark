"""Locate vendored prompt TOMLs by language pair + optional version pin."""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

_VERSION_RE = re.compile(r"v(\d+)\.(\d+)$")


def _version_key(stem: str) -> tuple[int, int] | None:
    """Parse `...vX.Y` from a TOML stem; return (X, Y) or None if not versioned."""
    match = _VERSION_RE.search(stem)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _prompt_dir(kind: str) -> Path:
    """Return the on-disk directory holding vendored prompts of `kind` (`align` or `evaluate`)."""
    pkg_dir = resources.files("kotoba_benchmark.prompts") / kind
    return Path(str(pkg_dir))


def find_prompt(
    kind: str,
    *,
    source_lang: str,
    target_lang: str,
    version: str | None = None,
) -> Path:
    """Resolve a prompt TOML file path for the requested language pair.

    Selection rules (when `version` is None):
      1. Prefer the highest-version pair-specific prompt: `{src}2{tgt}-vX.Y.toml`.
      2. Fall back to the highest-version pair-agnostic prompt: `vX.Y.toml`.

    When `version` is given (e.g. `"v1.2"` or `"1.2"`), select that exact version,
    pair-specific first then pair-agnostic. Raises `FileNotFoundError` if no match.
    """
    if kind not in {"align", "evaluate"}:
        raise ValueError(f"kind must be 'align' or 'evaluate', got {kind!r}")

    prompt_dir = _prompt_dir(kind)
    candidates = sorted(prompt_dir.glob("*.toml"))
    if not candidates:
        raise FileNotFoundError(f"No prompt files under {prompt_dir}")

    pair_prefix = f"{source_lang}2{target_lang}-"

    if version is not None:
        version = version if version.startswith("v") else f"v{version}"
        for c in candidates:
            if c.stem == f"{pair_prefix}{version}":
                return c
        for c in candidates:
            if c.stem == version:
                return c
        raise FileNotFoundError(
            f"No {kind} prompt found for pair={source_lang}2{target_lang} version={version}"
        )

    pair_specific = [c for c in candidates if c.stem.startswith(pair_prefix)]
    if pair_specific:
        ranked = sorted(
            pair_specific,
            key=lambda p: _version_key(p.stem) or (-1, -1),
        )
        return ranked[-1]

    generic = [c for c in candidates if _version_key(c.stem) is not None and "2" not in c.stem]
    if generic:
        ranked = sorted(generic, key=lambda p: _version_key(p.stem) or (-1, -1))
        return ranked[-1]

    raise FileNotFoundError(
        f"No {kind} prompt found for pair={source_lang}2{target_lang} "
        f"(searched {prompt_dir})"
    )
