from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multi_step_rag.config import load_yaml, set_by_dotted_path
from multi_step_rag.pipeline import run_pipeline


def run_experiments(experiments: list, grid: dict, base_config: dict, models: dict, 
                    processed_dir: Path, base_output_dir: Path, limit: int) -> list:
    """运行指定的实验列表"""
    summary_rows = []
    for dataset_name in grid["datasets"]:
        dataset_path = processed_dir / dataset_name
        for experiment in experiments:
            config = deepcopy(base_config)
            config["experiment_name"] = f"{experiment['name']}_{dataset_name.replace('.jsonl', '')}"
            config["dataset_path"] = str(dataset_path)
            config["output_dir"] = str(base_output_dir / config["experiment_name"])
            if limit > 0:
                config["runtime"]["limit"] = limit
            for key, value in experiment.items():
                if key == "name":
                    continue
                set_by_dotted_path(config, key, value)
            metrics = run_pipeline(config, models)
            summary_rows.append(
                {
                    "experiment": config["experiment_name"],
                    "dataset": dataset_name,
                    **metrics,
                }
            )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--processed_dir", default="")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--base_config", default="configs/base_experiment.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ablation", action="store_true", help="运行后 2 组消融实验 (exp09, exp10)")
    args = parser.parse_args()

    grid = load_yaml(args.grid)
    base_config = load_yaml(args.base_config)
    models = load_yaml(args.models)

    processed_dir = Path(args.processed_dir or grid.get("processed_dir", "data/processed"))
    base_output_dir = Path(grid.get("base_output_dir", "outputs/runs"))
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # 选择实验列表
    if args.ablation:
        # 运行后 2 组消融实验 (exp09, exp10)
        experiments = grid.get("ablation_experiments", [])
        print(f">>> 运行消融实验: {len(experiments)} 组")
    else:
        # 默认运行前 8 组消融实验
        experiments = grid.get("experiments", [])
        print(f">>> 运行主实验: {len(experiments)} 组")

    summary_rows = run_experiments(
        experiments, grid, base_config, models, 
        processed_dir, base_output_dir, args.limit
    )

    # 写入汇总 CSV
    fieldnames = sorted({key for row in summary_rows for key in row.keys()})
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f">>> 结果已保存至: {summary_path}")


if __name__ == "__main__":
    main()
