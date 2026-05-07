from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multi_step_rag.config import dump_yaml, load_yaml


def run_cmd(args: list[str]) -> None:
    print(">>>", " ".join(args))
    subprocess.run(args, check=True)


def build_eval_config(
    base_config_path: str,
    output_path: Path,
    model_key: str,
    dataset_path: str,
    output_dir: str,
    limit: int,
) -> None:
    config = load_yaml(base_config_path)
    config["dataset_path"] = dataset_path
    config["output_dir"] = output_dir
    config["experiment_name"] = Path(output_dir).name
    config["generation"]["model_key"] = model_key
    config["generation"]["override_model_path"] = ""
    if limit > 0:
        config["runtime"]["limit"] = limit
    dump_yaml(config, output_path)


def build_models_with_merged(
    base_models_path: str,
    output_path: Path,
    merged_model_key: str,
    merged_model_path: str,
) -> None:
    models = load_yaml(base_models_path)
    models[merged_model_key] = merged_model_path
    dump_yaml(models, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the weak-supervised SFT/LoRA pipeline end-to-end.")
    parser.add_argument("--predictions", nargs="+", required=True, help="One or more predictions.jsonl files for SFT data construction.")
    parser.add_argument("--sft_output", required=True, help="Output jsonl path for weakly supervised SFT data.")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--train_config", default="configs/lora_sft.yaml")
    parser.add_argument("--models", default="configs/model_paths.yaml")
    parser.add_argument("--base_model_key", default="generator_model")
    parser.add_argument("--adapter_output_dir", required=True)
    parser.add_argument("--merged_output_dir", required=True)
    parser.add_argument("--merged_model_key", default="lora_merged_model")
    parser.add_argument("--eval_base_config", default="configs/base_experiment.yaml")
    parser.add_argument("--eval_datasets", nargs="+", default=["data/processed/asqa_eval_top20.jsonl"])
    parser.add_argument("--eval_output_root", required=True)
    parser.add_argument("--eval_limit", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.eval_output_root)
    root.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            sys.executable,
            "scripts/build_sft_data.py",
            "--predictions",
            *args.predictions,
            "--output",
            args.sft_output,
            "--max_samples",
            str(args.max_samples),
        ]
    )

    run_cmd(
        [
            sys.executable,
            "scripts/train_sft.py",
            "--train_file",
            args.sft_output,
            "--config",
            args.train_config,
            "--models",
            args.models,
            "--model_key",
            args.base_model_key,
            "--output_dir",
            args.adapter_output_dir,
        ]
    )

    run_cmd(
        [
            sys.executable,
            "scripts/merge_lora.py",
            "--models",
            args.models,
            "--model_key",
            args.base_model_key,
            "--adapter_dir",
            str(Path(args.adapter_output_dir) / "final_model"),
            "--output_dir",
            args.merged_output_dir,
        ]
    )

    temp_models_path = root / "temp_models.merged.yaml"
    build_models_with_merged(args.models, temp_models_path, args.merged_model_key, args.merged_output_dir)

    for dataset_path in args.eval_datasets:
        dataset_name = Path(dataset_path).stem
        output_dir = root / f"{args.merged_model_key}_{dataset_name}"
        temp_config_path = root / f"temp_eval_{dataset_name}.yaml"
        build_eval_config(
            args.eval_base_config,
            temp_config_path,
            args.merged_model_key,
            dataset_path,
            str(output_dir),
            args.eval_limit,
        )
        run_cmd(
            [
                sys.executable,
                "scripts/run_pipeline.py",
                "--config",
                str(temp_config_path),
                "--models",
                str(temp_models_path),
            ]
            + (["--limit", str(args.eval_limit)] if args.eval_limit > 0 else [])
        )

    print(">>> Weak-supervised SFT experiment completed.")


if __name__ == "__main__":
    main()
