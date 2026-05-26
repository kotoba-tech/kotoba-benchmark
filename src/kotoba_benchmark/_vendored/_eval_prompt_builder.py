import json
from dataclasses import dataclass
from pathlib import Path

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
        prompt_path: Path | str,
    ):
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

        self.source_lang: str = self.raw_content.get("source_lang")
        self.target_lang: str = self.raw_content.get("target_lang")

        assert isinstance(self.source_lang, str) and isinstance(
            self.target_lang, str
        ), "Source and target languages must be specified and str in the prompt file."

        self.fewshot_input_str = self.create_fewshot_input_str(self.fewshot_examples)
        self.fewshot_output_str = self.create_fewshot_output_str(self.fewshot_examples)

    def create_fewshot_input_str(self, fewshot_examples: list[dict]) -> str:
        input_examples: dict[str, list[dict]] = {
            "input": [
                {
                    self.source_lang: example[self.source_lang],
                    self.target_lang: example[self.target_lang],
                }
                for example in fewshot_examples
            ]
        }
        return json.dumps(input_examples, ensure_ascii=False, indent=4)

    def create_fewshot_output_str(self, fewshot_examples: list[dict]) -> str:
        output_examples: dict[str, list[dict]] = {
            "output": [
                {
                    self.source_lang: example[self.source_lang],
                    self.target_lang: example[self.target_lang],
                    "fluency": example["fluency"],
                    "accuracy": example["accuracy"],
                    "conciseness": example["conciseness"],
                }
                for example in fewshot_examples
            ]
        }
        return json.dumps(output_examples, ensure_ascii=False, indent=4)

    def create_input_str(
        self,
        source_texts: list[str],
        target_texts: list[str],
    ) -> str:
        input_examples: dict[str, list[dict]] = {
            "input": [
                {
                    self.source_lang: source_text,
                    self.target_lang: target_text,
                }
                for source_text, target_text in zip(source_texts, target_texts)
            ]
        }
        return json.dumps(input_examples, ensure_ascii=False, indent=4)

    def fill_prompt_template(
        self,
        input_str: str,
    ):
        content: str = self.prompt_template.format(
            input_str=input_str,
            fewshot_input_str=self.fewshot_input_str,
            fewshot_output_str=self.fewshot_output_str,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
        )
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
        source_texts: list[str],
        target_texts: list[str],
    ) -> Prompt:
        input_str = self.create_input_str(source_texts, target_texts)
        filled_template = self.fill_prompt_template(input_str)
        messages = self.create_messages(filled_template)

        return Prompt(
            processor_version=self.processor_version,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            messages=messages,
        )
