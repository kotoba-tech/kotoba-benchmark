import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli as toml


@dataclass
class Prompt:
    processor_version: str
    source_lang: str
    target_lang: str
    messages: list[dict]


class PromptBuilder:
    prompt_path: Path
    processor_version: str
    source_lang: str
    target_lang: str
    fewshot_examples: list[dict]

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        prompt_path: Path | str,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        assert isinstance(self.source_lang, str) and isinstance(
            self.target_lang, str
        ), "Source and target languages must be specified and str in the prompt file."

        self.prompt_path = Path(prompt_path)
        self.prompt_template: str = self.prompt_path.read_text(encoding="utf-8")
        self.raw_content = toml.loads(self.prompt_template)

        self.processor_version: str = self.raw_content.get("processor_version")
        assert isinstance(self.processor_version, str), (
            "Processor version must be specified and str in the prompt file."
        )

        self.fewshot_examples = self.raw_content.get("fewshot_examples", [])
        assert isinstance(self.fewshot_examples, list) and all(
            isinstance(example, dict) for example in self.fewshot_examples
        ), "Fewshot examples must be a list of dicts in the prompt file."

    @staticmethod
    def _normalize_alignment_input(value: Any) -> tuple[str, str]:
        if isinstance(value, str):
            return value, value
        if not isinstance(value, list):
            raise TypeError(
                "Alignment input must be a string or a list of {start,end,text} items."
            )

        normalized: list[dict[str, float | str]] = []
        text_parts: list[str] = []
        for idx, item in enumerate(value):
            if not isinstance(item, dict):
                raise TypeError(
                    f"Alignment input item {idx} must be a dict, got {type(item).__name__}"
                )
            start = item.get("start")
            end = item.get("end")
            text = item.get("text")
            if text is None and "word" in item:
                text = item.get("word")
            if start is None or end is None or not isinstance(text, str):
                raise ValueError(
                    "Alignment input items must include start, end, and text/word."
                )
            normalized.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "text": text,
                }
            )
            text_parts.append(text)

        prompt_value = json.dumps(normalized, ensure_ascii=False, indent=2)
        text_value = "".join(text_parts)
        return prompt_value, text_value

    def _build_prompt_variables(
        self,
        source_input: Any,
        target_input: Any,
    ) -> dict[str, Any]:
        """Build format variables for one alignment prompt."""
        source_input_json, source_text = self._normalize_alignment_input(source_input)
        target_input_json, target_text = self._normalize_alignment_input(target_input)

        return {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "source_text": source_text,
            "target_text": target_text,
            "source_input": source_input_json,
            "target_input": target_input_json,
            "source_segments_json": source_input_json,
            "target_segments_json": target_input_json,
            "source_timestamps_json": source_input_json,
            "target_timestamps_json": target_input_json,
        }

    def fill_prompt_template(
        self,
        source_input: Any,
        target_input: Any,
    ):
        template_values = self._build_prompt_variables(
            source_input=source_input,
            target_input=target_input,
        )
        content: str = self.prompt_template.format(**template_values)
        return content

    def create_messages(
        self,
        content: str,
    ) -> list[dict]:
        messages: list[dict] = toml.loads(content)["messages"]
        messages = [
            {**messages, "content": messages["content"].strip()}
            for messages in messages
        ]
        return messages

    def build_prompt(
        self,
        source_input: Any,
        target_input: Any,
    ) -> Prompt:
        filled_template = self.fill_prompt_template(
            source_input=source_input,
            target_input=target_input,
        )
        messages = self.create_messages(filled_template)

        return Prompt(
            processor_version=self.processor_version,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            messages=messages,
        )
