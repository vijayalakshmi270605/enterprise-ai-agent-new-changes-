import os
import json
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, PeftModel
from app.config import settings
from training.dataset import InstructionDataset

logger = logging.getLogger(__name__)


class LoRATrainer:
    def __init__(self, train_path: str, output_dir: str = None):
        self.train_path = train_path
        self.output_dir = output_dir or settings.lora_output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_base,
            trust_remote_code=True,
            truncation_side="left"
        )
        self.dataset_helper = InstructionDataset(self.tokenizer, max_length=settings.max_tokens)

    def prepare_dataset(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        with open(self.train_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
            
        processed = []
        for ex in examples:
            prompt = ex.get("instruction", "")
            if ex.get("input"):
                prompt += "\n" + ex["input"]
                
            system_prompt = "You are a highly capable AI assistant, similar to ChatGPT or Google Assistant.\n\nYour absolute highest priority is FACTUAL ACCURACY. DO NOT HALLUCINATE or make up information.\n\nWhen answering, STRICTLY base your responses on the Retrieved Knowledge or Tool results provided below.\n\nIf the retrieved context does not contain the answer, explicitly state 'I do not have enough information to answer that' instead of guessing.\n\nProvide clear, direct, and well-structured answers."
            
            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            messages = prompt_messages + [{"role": "assistant", "content": ex.get("output", "")}]
            
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            encoded = self.tokenizer(text, truncation=True, max_length=settings.max_tokens, padding="max_length")
            
            # Calculate the length of the prompt to mask it in the labels
            prompt_text = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            prompt_encoded = self.tokenizer(prompt_text, truncation=True, max_length=settings.max_tokens)
            prompt_len = len(prompt_encoded["input_ids"])
            
            input_ids = torch.tensor(encoded["input_ids"])
            attention_mask = torch.tensor(encoded["attention_mask"])
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            # CRITICAL FIX: Mask the prompt tokens so the model ONLY trains on the generated answer
            labels[:min(prompt_len, len(labels))] = -100
            
            processed.append({
                "input_ids": input_ids.tolist(),
                "attention_mask": attention_mask.tolist(),
                "labels": labels.tolist()
            })
            
        return processed

    def configure_peft(self):
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return peft_config

    def train(self):
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_base,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        if os.path.exists(self.output_dir) and os.path.exists(os.path.join(self.output_dir, "adapter_config.json")):
            logger.info("Loading existing LoRA adapter from %s to resume training", self.output_dir)
            peft_model = PeftModel.from_pretrained(model, self.output_dir, is_trainable=True)
        else:
            logger.info("Initializing new LoRA adapter")
            peft_model = get_peft_model(model, self.configure_peft())

        peft_model.train()
        peft_model.print_trainable_parameters()

        train_dataset = self.prepare_dataset()
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=settings.fine_tune_learning_rate)
        device = peft_model.device if hasattr(peft_model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            peft_model.to(device)

        max_steps = int(getattr(settings, "fine_tune_max_steps", 8))
        for step in range(max_steps):
            record = train_dataset[step % len(train_dataset)]
            batch = {
                key: torch.tensor([value], device=device)
                for key, value in record.items()
                if key in {"input_ids", "attention_mask", "labels"}
            }
            outputs = peft_model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            logger.info("Fine-tune step %d/%d loss=%.4f", step + 1, max_steps, loss.item())
            print(f"fine-tune step {step + 1}/{max_steps} loss={loss.item():.4f}", flush=True)

        peft_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info("Saved LoRA adapter to %s", self.output_dir)
