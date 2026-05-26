"""Pluggable translate-stage backends.

The translate stage is the only part of the pipeline tied to a specific
STS vendor. Backends implement `TranslateBackend` and register themselves
via `@register("name")`. Built-in backends live in this package; users
can plug in their own by passing an instance to `Config(translate_backend=...)`
or by referencing an importable class in TOML (`backend = "pkg.module.MyBackend"`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass
class TranslateChunk:
    """One unit of translated output streamed from the backend.

    `audio` is PCM16 mono at `sample_rate`. `received_at_monotonic` is the local
    wall-clock arrival time used for latency metrics. Either `partial_source` or
    `audio` may be populated on a given chunk; both empty signals end-of-stream.
    """

    audio: bytes
    sample_rate: int
    partial_source: str | None
    received_at_monotonic: float


class TranslateBackend(Protocol):
    """Backend contract: PCM16 in → translated audio + partial source transcripts out.

    `pcm16` is little-endian mono PCM16 at `sample_rate` Hz. Implementations
    yield `TranslateChunk`s as the server emits output; the iterator should
    end when the server signals end-of-turn.
    """

    async def translate(
        self,
        *,
        pcm16: bytes,
        sample_rate: int,
        source_lang: str,
        target_lang: str,
    ) -> AsyncIterator[TranslateChunk]: ...


BACKENDS: dict[str, type[TranslateBackend]] = {}


def register(name: str):
    def deco(cls: type[TranslateBackend]) -> type[TranslateBackend]:
        BACKENDS[name] = cls
        return cls

    return deco


def get_backend(name_or_class: str | type[TranslateBackend]) -> type[TranslateBackend]:
    """Resolve a backend by registry name or by importable dotted path."""
    if not isinstance(name_or_class, str):
        return name_or_class
    if name_or_class in BACKENDS:
        return BACKENDS[name_or_class]
    if "." in name_or_class:
        import importlib

        module_path, class_name = name_or_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    raise KeyError(
        f"Unknown translate backend {name_or_class!r}. "
        f"Built-in: {sorted(BACKENDS)}. For a custom backend, use a dotted path like 'mypkg.MyBackend'."
    )
