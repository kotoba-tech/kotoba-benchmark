# Docs

Read in this order if you're new:

| Doc | What it covers |
|---|---|
| [`quickstart.md`](quickstart.md) | Install → configure → run your first eval. The fastest path from `pip install` to a translated-audio + scored-clips HTML report. |
| [`config-reference.md`](config-reference.md) | Every field of the run config (TOML or Python), what it does, and its default. Use this to look up a flag. |
| [`metrics.md`](metrics.md) | What each number in the summary means — accuracy / fluency / conciseness scoring, translation-delay latency, coverage diagnostics. |
| [`extending.md`](extending.md) | Plug in a non-Kotoba STS backend (OpenAI Realtime, in-house servers, etc.) by implementing the `TranslateBackend` protocol. |

The CLI's own `--help` is also a reference:

```bash
kotoba-benchmark --help
kotoba-benchmark run --help
kotoba-benchmark report --help
kotoba-benchmark show-prompts --help
```
