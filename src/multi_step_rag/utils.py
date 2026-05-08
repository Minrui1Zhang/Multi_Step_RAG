from __future__ import annotations

import json
import random
import re
import string
from pathlib import Path


CITATION_RE = re.compile(r"\[(\d+)\]")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def simple_tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    try:
        from nltk.tokenize import sent_tokenize

        return [sent.strip() for sent in sent_tokenize(text) if sent.strip()]
    except Exception:
        pieces = re.split(r"(?<=[.!?])\s+", text)
        return [piece.strip() for piece in pieces if piece.strip()]


def remove_citations(text: str) -> str:
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    text = re.sub(r"\[\d+\](?:\[\d+\])+", "", text)
    return " ".join(text.split())


def extract_citation_ids(text: str) -> list[int]:
    return [int(match) for match in CITATION_RE.findall(text)]


def batched(items: list, batch_size: int) -> list[list]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
