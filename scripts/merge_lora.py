from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from multi_step_rag.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model.")
    parser.add_argument("--models", required=True)
    parser.add_argument("--model_key", default="generator_model")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    model_paths = load_yaml(args.models)
    base_model_path = model_paths[args.model_key]

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    merged_model = peft_model.merge_and_unload()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f">>> Merged model saved to: {output_dir}")


if __name__ == "__main__":
    main()
