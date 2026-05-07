from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multi_step_rag.sft import save_sft_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+",required=True)  #added
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=2000)
    args = parser.parse_args()

    save_sft_rows(args.predictions, args.output, args.max_samples)


if __name__ == "__main__":
    main()
