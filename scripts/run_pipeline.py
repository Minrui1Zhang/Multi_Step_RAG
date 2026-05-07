from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multi_step_rag.config import dump_yaml, load_yaml
from multi_step_rag.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = load_yaml(args.config)
    models = load_yaml(args.models)
    if args.dataset:
        config["dataset_path"] = args.dataset
    if args.output:
        config["output_dir"] = args.output
    if args.limit > 0:
        config["runtime"]["limit"] = args.limit

    run_pipeline(config, models)


if __name__ == "__main__":
    main()
