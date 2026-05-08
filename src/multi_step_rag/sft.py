from __future__ import annotations

from .generation import build_prompt
from .utils import load_jsonl, save_jsonl


def _collect_prediction_rows(predictions_paths: list[str]) -> list[dict]:
    rows = []
    for path in predictions_paths:
        rows.extend(load_jsonl(path))
    return rows


def build_sft_rows(predictions_paths: list[str], max_samples: int) -> list[dict]:
    rows = _collect_prediction_rows(predictions_paths)
    output = []
    for row in rows[:max_samples]:
        corrected = row.get("reflected_answer") or row.get("prediction")
        prompt = build_prompt(
            {"question": row["question"]},
            {"docs": row["retrieved_docs"]},
            row.get("extracted", []),
        )
        output.append(
            {
                "id": row["id"],
                "prompt": prompt,
                "response": corrected,
                "dataset": row.get("dataset"),
                "trace": row.get("reflection_trace", []),
            }
        )
    return output


def save_sft_rows(predictions_paths: list[str], output_path: str, max_samples: int) -> None:
    rows = build_sft_rows(predictions_paths, max_samples)
    save_jsonl(rows, output_path)
