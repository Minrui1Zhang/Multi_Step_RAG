from __future__ import annotations

from typing import Any
import warnings

from .utils import extract_citation_ids, remove_citations, simple_tokens, split_sentences


def _resolve_dtype(name: str) -> Any:
    import torch

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return "auto"


class EntailmentScorer:
    def __init__(self, model_path: str, dtype: str, device: str) -> None:
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.model_kind = ""
        self.model = None
        self.tokenizer = None
        self.entailment_label_id = None

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        load_kwargs = {
            "dtype": _resolve_dtype(self.dtype),
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        if getattr(config, "is_encoder_decoder", False):
            self.model_kind = "seq2seq"
            #self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, **load_kwargs)
            self.model = self._load_model(AutoModelForSeq2SeqLM, load_kwargs) # new added
            return

        self.model_kind = "sequence_classification"
        #self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, **load_kwargs)
        self.model = self._load_model(AutoModelForSequenceClassification, load_kwargs)
        self.entailment_label_id = self._find_entailment_label_id()

    def _load_model(self, model_cls, load_kwargs: dict):
        try:
            return model_cls.from_pretrained(self.model_path, **load_kwargs)
        except Exception as exc:
            message = str(exc).lower()
            if "safetensor" not in message and "safetensors" not in message:
                raise
            warnings.warn(
                f"Safetensors load failed for {self.model_path}; retrying with use_safetensors=False.",
                RuntimeWarning,
            )
            return model_cls.from_pretrained(
                self.model_path,
                use_safetensors=False,
                **load_kwargs,
            )

    def _find_entailment_label_id(self) -> int:
        id2label = getattr(self.model.config, "id2label", {}) or {}
        for label_id, label in id2label.items():
            if "entail" in str(label).lower():
                return int(label_id)
        if len(id2label) == 2:
            warnings.warn(
                f"No explicit entailment label found for {self.model_path}; using the positive label id.",
                RuntimeWarning,
            )
            return max(int(label_id) for label_id in id2label)
        raise ValueError(
            f"Could not identify an entailment label for {self.model_path}. "
            f"Model id2label={id2label!r}"
        )

    def _model_device(self):
        return getattr(self.model, "device", None) or next(self.model.parameters()).device

    def _score_sequence_classification(self, premise: str, hypothesis: str) -> float:
        import torch

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs = {key: value.to(self._model_device()) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits.float(), dim=-1)
        return float(probs[int(self.entailment_label_id)].item())

    def _score_seq2seq(self, premise: str, hypothesis: str) -> float:
        prompt = f"premise: {premise} hypothesis: {hypothesis}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {key: value.to(self._model_device()) for key, value in inputs.items()}
        output_ids = self.model.generate(**inputs, max_new_tokens=8)
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()
        if decoded in {"1", "entailment", "entailed", "true", "yes"}:
            return 1.0
        if decoded in {"0", "contradiction", "contradict", "neutral", "not_entailment", "false", "no"}:
            return 0.0
        if "entail" in decoded or "true" in decoded or "yes" in decoded:
            return 1.0
        if "contrad" in decoded or "neutral" in decoded or "false" in decoded or "no" in decoded:
            return 0.0
        raise ValueError(f"Unrecognized NLI seq2seq output from {self.model_path}: {decoded!r}")

    def score(self, premise: str, hypothesis: str) -> float:
        premise = premise.strip()
        hypothesis = hypothesis.strip()
        if not premise or not hypothesis:
            return 0.0
        self.ensure_loaded()
        if self.model_kind == "sequence_classification":
            return self._score_sequence_classification(premise, hypothesis)
        if self.model_kind == "seq2seq":
            return self._score_seq2seq(premise, hypothesis)
        raise RuntimeError(f"Unsupported NLI model kind: {self.model_kind!r}")


def filter_sentences(question: str, retrieval_output: dict, scorer: EntailmentScorer | None, cfg: dict) -> dict:
    mode = cfg.get("mode", "none")
    sentences = retrieval_output.get("sentences", [])
    if mode == "none":
        return retrieval_output

    ranked = []
    for row in sentences:
        if mode == "overlap":
            q = set(simple_tokens(question))
            s = set(simple_tokens(row["sentence"]))
            score = len(q & s) / max(len(q), 1)
        else:
            score = scorer.score(row["sentence"], question) if scorer is not None else 0.0
        ranked.append({**row, "filter_score": score})

    threshold = float(cfg.get("threshold", 0.0))
    keep_topk = int(cfg.get("keep_topk", len(ranked)))
    kept = [row for row in ranked if row["filter_score"] >= threshold]
    if not kept:
        kept = ranked
    kept.sort(key=lambda item: (item["filter_score"], item["score"]), reverse=True)
    return {
        **retrieval_output,
        "sentences": kept[:keep_topk],
    }


def _doc_map(retrieval_output: dict) -> dict[int, dict]:
    return {doc["doc_id"]: doc for doc in retrieval_output.get("docs", [])}


def reflect_answer(answer: str, retrieval_output: dict, extracted: list[dict], scorer: EntailmentScorer | None, cfg: dict) -> dict:
    mode = cfg.get("mode", "none")
    if mode == "none":
        return {
            "answer": answer,
            "reflection_trace": [],
        }

    docs = _doc_map(retrieval_output)
    sentences = split_sentences(answer)
    trace = []
    revised_sentences = []
    threshold = float(cfg.get("threshold", 0.55))
    max_fix = int(cfg.get("max_fix_sentences", 3))
    fixed_count = 0

    fallback_facts = [item for item in extracted if item["type"] == "fact"]
    fallback_idx = 0

    for sent in sentences:
        citation_ids = extract_citation_ids(sent)
        clean_sent = remove_citations(sent)
        support_text = " ".join(docs[cid]["text"] for cid in citation_ids if cid in docs)
        if not citation_ids:
            support_score = 0.0
        elif mode == "rule":
            support_score = 1.0 if support_text else 0.0
        else:
            support_score = scorer.score(support_text, clean_sent) if scorer is not None else 0.0

        supported = support_score >= threshold
        entry = {
            "sentence": sent,
            "citation_ids": citation_ids,
            "support_score": round(float(support_score), 4),
            "supported": supported,
        }

        if supported or fixed_count >= max_fix:
            revised_sentences.append(sent)
            trace.append(entry)
            continue

        # Replace unsupported sentences with the highest-ranked extracted facts.
        # This is the core "attempt -> reflection -> correction" behavior chain.
        if fallback_idx < len(fallback_facts):
            replacement = f"{fallback_facts[fallback_idx]['fact']} {fallback_facts[fallback_idx]['citation']}"
            fallback_idx += 1
        elif retrieval_output.get("sentences"):
            best = retrieval_output["sentences"][0]
            replacement = f"{best['sentence']} [{best['doc_id']}]"
        else:
            replacement = sent

        fixed_count += 1
        trace.append({**entry, "replacement": replacement})
        revised_sentences.append(replacement)

    return {
        "answer": " ".join(revised_sentences).strip(),
        "reflection_trace": trace,
    }
