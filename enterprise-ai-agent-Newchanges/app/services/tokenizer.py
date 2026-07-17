import logging
from transformers import AutoTokenizer
from app.config import settings

logger = logging.getLogger(__name__)


class TokenizerPipeline:
    def __init__(self):
        self.tokenizer = None

    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_base, trust_remote_code=True)
        logger.info("Instruction tokenizer initialized for %s", settings.model_base)

    def encode_instruction(self, instruction: str, input_text: str = ""):
        prompt = """
<|user|>
{instruction}
<|assistant|>
{input}
""".format(instruction=instruction, input=input_text)
        return self.tokenizer(prompt, truncation=True, padding="max_length", max_length=settings.max_tokens)

    def format_examples(self, examples):
        return [
            self.encode_instruction(example["instruction"], example.get("input", ""))
            for example in examples
        ]


tokenizer_pipeline = TokenizerPipeline()
