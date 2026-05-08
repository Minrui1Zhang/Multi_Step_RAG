from __future__ import annotations

from pathlib import Path

from .utils import load_json, save_jsonl


def _asqa_target(record: dict) -> str:
    annotations = record.get("annotations") or []
    if len(annotations) >= 2:
        return annotations[1].get("long_answer", "")
    if annotations:
        return annotations[0].get("long_answer", "")
    return ""


def _qampari_target(record: dict) -> str:
    answers = []
    for item in record.get("answers", []):
        if item:
            answers.append(item[0])
    return ", ".join(answers)


def _eli5_target(record: dict) -> str:
    return record.get("answer", "")


def normalize_alce_record(record: dict, dataset_name: str, topk_docs: int) -> dict:
    docs = []
    for doc_id, doc in enumerate(record.get("docs", [])[:topk_docs], start=1):
        docs.append(
            {
                "doc_id": doc_id,
                "title": doc.get("title", f"Doc {doc_id}"),
                "text": doc.get("text", ""),
                "summary": doc.get("summary", ""),
                "extraction": doc.get("extraction", ""),
                "url": doc.get("url", ""),
                "score": float(doc.get("score", 0.0)),
            }
        )

    target = ""
    if dataset_name == "asqa":
        target = _asqa_target(record)
    elif dataset_name == "qampari":
        target = _qampari_target(record)
    else:
        target = _eli5_target(record)

    return {
        "id": record.get("id", record.get("question", "")[:48]),
        "dataset": dataset_name,
        "question": record["question"],
        "target": target,
        "docs": docs,
        "qa_pairs": record.get("qa_pairs"),
        "answers": record.get("answers"),
        "claims": record.get("claims"),
        "annotations": record.get("annotations"),
        "question_ctx": record.get("question_ctx"),
        "metadata": {
            "num_docs": len(docs),
        },
    }


def prepare_alce_directory(input_dir: str | Path, output_dir: str | Path, topk_docs: int) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "asqa_eval_gtr_top100.json": ("asqa", "asqa_eval_top20.jsonl"),
        "eli5_eval_bm25_top100.json": ("eli5", "eli5_eval_top20.jsonl"),
        "qampari_eval_gtr_top100.json": ("qampari", "qampari_eval_top20.jsonl"),
    }

    for filename, (dataset_name, out_name) in mapping.items():
        src = input_dir / filename
        if not src.exists():
            continue
        raw = load_json(src)
        rows = [normalize_alce_record(item, dataset_name, topk_docs) for item in raw]
        save_jsonl(rows, output_dir / out_name)
