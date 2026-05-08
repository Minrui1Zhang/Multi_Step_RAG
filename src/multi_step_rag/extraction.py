from __future__ import annotations

from .utils import normalize_answer


def _deduplicate(items: list[dict], key: str) -> list[dict]:
    seen = set()
    output = []
    for item in items:
        marker = normalize_answer(item.get(key, ""))
        if not marker or marker in seen:
            continue
        seen.add(marker)
        output.append(item)
    return output


def extract_notes(sentences: list[dict], max_items: int) -> list[dict]:
    notes = []
    for row in sentences[:max_items]:
        note_text = (row.get("local_context") or row["sentence"]).strip()
        notes.append(
            {
                "type": "note",
                "text": note_text,
                "focus": row["sentence"],
                "citation": f"[{row['doc_id']}]",
                "title": row["title"],
                "score": round(row["score"], 4),
            }
        )
    return notes


def extract_facts(sentences: list[dict], max_items: int) -> list[dict]:
    facts = []
    for rank, row in enumerate(sentences[:max_items], start=1):
        support_text = (row.get("local_context") or row["sentence"]).strip() #new added
        facts.append(
            {
                "type": "fact",
                "fact": row["sentence"],
                "support": support_text,
                "citation": f"[{row['doc_id']}]",
                "title": row["title"],
                "rank": rank,
                "score": round(row["score"], 4),
            }
        )
    return facts


def run_extraction(retrieval_output: dict, cfg: dict) -> list[dict]:
    mode = cfg.get("mode", "none")
    max_items = int(cfg.get("max_items", 8))
    sentences = retrieval_output.get("sentences", [])

    if mode == "none":
        return []
    if mode == "notes":
        items = extract_notes(sentences, max_items)
    elif mode == "facts":
        items = extract_facts(sentences, max_items)
    else:
        raise ValueError(f"Unsupported extraction mode: {mode}")
    
    if cfg.get("deduplicate", True):
        key = "fact" if mode == "facts" else "text"
        items = _deduplicate(items, key)
    return items[:max_items]
