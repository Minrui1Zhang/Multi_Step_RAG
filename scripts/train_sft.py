from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from multi_step_rag.config import load_yaml
from multi_step_rag.utils import load_jsonl


def build_dataset(rows: list[dict], tokenizer, max_length: int) -> Dataset:
    def _tokenize(example: dict) -> dict:
        text = example["prompt"] + example["response"]
        tokenized = tokenizer(text, truncation=True, max_length=max_length)
        tokenized["labels"] = tokenized["input_ids"][:]
        return tokenized

    return Dataset.from_list(rows).map(_tokenize, remove_columns=list(rows[0].keys()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_key", default="generator_model")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_paths = load_yaml(args.models)
    rows = load_jsonl(args.train_file)
    rows = [row for row in rows if row.get("prompt") and row.get("response")]

    model_path = model_paths[args.model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    peft_cfg = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        target_modules=cfg.get("target_modules", []),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    train_dataset = build_dataset(rows, tokenizer, int(cfg.get("max_length", 2048)))
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    warmup_ratio = float(cfg.get("warmup_ratio", 0.05))
    warmup_steps = int(cfg.get("warmup_steps", 0))
    if warmup_steps <= 0 and warmup_ratio > 0:
        warmup_steps = max(1, int(int(cfg.get("max_steps", 800)) * warmup_ratio))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=int(cfg.get("max_steps", 800)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        warmup_steps=warmup_steps,
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 100)),
        report_to=[],
        bf16=True,
        remove_unused_columns=False,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": collator,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(Path(args.output_dir) / "final_model"))
    tokenizer.save_pretrained(str(Path(args.output_dir) / "final_model"))


if __name__ == "__main__":
    main()
