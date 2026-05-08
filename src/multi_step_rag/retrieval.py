from __future__ import annotations

import math
from collections import Counter

from .utils import simple_tokens, split_sentences


def bm25_like_score(query: str, text: str, avgdl: float = 100.0) -> float:
    q_tokens = simple_tokens(query)
    d_tokens = simple_tokens(text)
    if not q_tokens or not d_tokens:
        return 0.0
    counts = Counter(d_tokens)
    k1 = 1.2
    b = 0.75
    score = 0.0
    dl = max(len(d_tokens), 1)
    for token in q_tokens:
        tf = counts.get(token, 0)
        if tf == 0:
            continue
        denom = tf + k1 * (1 - b + b * dl / avgdl)
        score += tf * (k1 + 1) / denom
    return score


def score_doc(question: str, doc: dict, cfg: dict) -> float:
    title_boost = float(cfg.get("title_boost", 0.0))
    summary_boost = float(cfg.get("summary_boost", 0.0))
    extraction_boost = float(cfg.get("extraction_boost", 0.0))
    text_score = bm25_like_score(question, doc["text"])
    title_score = bm25_like_score(question, doc["title"])
    summary_score = bm25_like_score(question, doc.get("summary", "")) if cfg.get("use_summary", True) else 0.0
    extraction_score = bm25_like_score(question, doc.get("extraction", "")) if cfg.get("use_extraction", True) else 0.0
    prior = math.log1p(max(doc.get("score", 0.0), 0.0))
    return text_score + title_boost * title_score + summary_boost * summary_score + extraction_boost * extraction_score + 0.05 * prior


def score_sentence(question: str, sentence: str, title: str, local_context: str) -> float:
    return (
        bm25_like_score(question, sentence)
        + 0.25 * bm25_like_score(question, local_context)
        + 0.15 * bm25_like_score(question, title)
    )


def rerank(example: dict, cfg: dict) -> dict:
    doc_topk = int(cfg.get("doc_topk", 5))
    sentence_topk = int(cfg.get("sentence_topk", 10))
    window = int(cfg.get("sentence_context_window", 1))
    question = example["question"]

    docs = [{**doc, "rerank_score": score_doc(question, doc, cfg)} for doc in example["docs"]]
    docs.sort(key=lambda item: item["rerank_score"], reverse=True)
    top_docs = docs[:doc_topk]

    sentence_candidates = []
    for doc in top_docs:
        sentences = split_sentences(doc["text"])
        if not sentences:
            sentences = [doc["text"]]
        for sent_idx, sentence in enumerate(sentences):
            left = max(0, sent_idx - window)
            right = sent_idx + window + 1
            local_context = " ".join(sentences[left:right])
            sentence_candidates.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "sentence_id": sent_idx,
                    "sentence": sentence,
                    "local_context": local_context,
                    "score": score_sentence(question, sentence, doc["title"], local_context),
                }
            )

    sentence_candidates.sort(key=lambda item: item["score"], reverse=True)
    selected_sentences = sentence_candidates[:sentence_topk] if sentence_topk > 0 else []

    return {
        "docs": top_docs,
        "sentences": selected_sentences,
    }
