import json
from pathlib import Path
from typing import List, Dict
from transformers import PreTrainedTokenizerBase


class InstructionDataset:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def load_json(path: str) -> List[Dict[str, str]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def build_instruction_prompt(self, instruction: str, input_text: str = "") -> str:
        user_text = instruction if not input_text else f"{instruction}\n\n{input_text}"
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        template = (
            "<|user|>\n"
            "{instruction}\n"
            "<|assistant|>\n"
            "{input_text}"
        )
        return template.format(instruction=instruction, input_text=input_text)

    def tokenize_example(self, example: Dict[str, str]) -> Dict[str, List[int]]:
        prompt = self.build_instruction_prompt(example["instruction"], example.get("input", ""))
        target = example.get("output", "")
        full_text = f"{prompt}{target}"
        tokens = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    def prepare_dataset(self, examples: List[Dict[str, str]]):
        return [self.tokenize_example(ex) for ex in examples]

    def save_instruction_dataset(self, examples: List[Dict[str, str]], output_path: str):
        output = [
            {
                "instruction": ex["instruction"],
                "input": ex.get("input", ""),
                "output": ex.get("output", ""),
            }
            for ex in examples
        ]
        Path(output_path).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
