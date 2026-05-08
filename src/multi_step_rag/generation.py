from __future__ import annotations

from typing import Any
import warnings


def _resolve_dtype(name: str) -> Any:
    import torch

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return "auto"


class OfflineGenerator:
    def __init__(self, model_path: str, dtype: str, device: str, max_new_tokens: int, temperature: float) -> None:
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self.uses_chat_template = False

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.uses_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        load_kwargs = {
            "torch_dtype": _resolve_dtype(self.dtype),
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        except Exception as exc:
            message = str(exc).lower()
            if "safetensor" not in message and "safetensors" not in message:
                raise
            warnings.warn(
                f"Safetensors load failed for {self.model_path}; retrying with use_safetensors=False.",
                RuntimeWarning,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                use_safetensors=False,
                **load_kwargs,
            )

    def _model_device(self):
        return getattr(self.model, "device", None) or next(self.model.parameters()).device

    def _build_model_inputs(self, prompt: str):
        if self.uses_chat_template:
            messages = [
                {
                    "role": "system",
                    "content": "You are a careful long-form QA assistant. Follow citation instructions exactly.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            rendered = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(rendered, "to") and hasattr(rendered, "input_ids"):
                return rendered
            return {"input_ids": rendered}
        return self.tokenizer(prompt, return_tensors="pt")

    def generate(self, prompt: str) -> str:
        self.ensure_loaded()
        model_inputs = self._build_model_inputs(prompt)
        model_device = self._model_device()
        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(model_device)
        else:
            model_inputs = {key: value.to(model_device) for key, value in model_inputs.items()}

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        generate_kwargs = {
            **model_inputs,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature

        output_ids = self.model.generate(**generate_kwargs)
        new_tokens = output_ids[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_prompt(example: dict, retrieval_output: dict, extracted: list[dict]) -> str:
    doc_lines = []
    for doc in retrieval_output.get("docs", []):
        preview = doc.get("summary") or doc.get("text", "")[:300]
        doc_lines.append(f"[{doc['doc_id']}] {doc['title']}: {preview}")

    sentence_lines = []
    for row in retrieval_output.get("sentences", []):
        sentence_lines.append(f"- [{row['doc_id']}] {row['sentence']}")

    extracted_mode = extracted[0]["type"] if extracted else "none"
    evidence_sections = []
    if extracted_mode == "fact":
        fact_lines = []
        for item in extracted:
            fact_lines.append(f"- Fact {item.get('rank', 0)}: {item['fact']} {item['citation']}")
            support = item.get("support", "").strip()
            if support and support != item["fact"]:
                fact_lines.append(f"  Support: {support}")
        evidence_sections.append(
            (
                "Candidate Facts are concise summaries of the reranked evidence. "
                "Use them as answer hints, but verify details against the supporting evidence sentences and documents.\n",
                "Candidate Facts:\n",
                fact_lines,
            )
        )
    elif extracted_mode == "note":
        note_lines = []
        for item in extracted:
            title = item.get("title", "")
            focus = item.get("focus", "")
            note_lines.append(f"- Note {item['citation']} ({title}): {item['text']}")
            if focus and focus != item["text"]:
                note_lines.append(f"  Focus sentence: {focus}")
        evidence_sections.append(
            (
                "Evidence Notes are raw contextual snippets. Compare them carefully and infer the answer conservatively "
                "instead of treating every note as a final fact.\n",
                "Evidence Notes:\n",
                note_lines,
            )
        )
    if sentence_lines:
        evidence_sections.append(
            (
                "Evidence Sentences are the reranked sentence-level snippets selected from the top documents. "
                "Use them to ground the answer and resolve ambiguity in any extracted notes or facts.\n",
                "Evidence Sentences:\n",
                sentence_lines,
            )
        )

    prompt = (
        "You are a careful long-form QA assistant.\n"
        "Answer the question only with support from the provided documents.\n"
        "Every major factual sentence must include one or more citations in bracket form like [1] or [1][3].\n"
        "Do not invent facts beyond the evidence. If evidence is weak, say the uncertainty explicitly.\n\n"
        f"Question: {example['question']}\n\n"
        "Top Documents:\n"
        + "\n".join(doc_lines)
    )
    for guidance, header, lines in evidence_sections:
        if lines:
            prompt += "\n\n" + guidance + "\n" + header + "\n".join(lines)
    prompt += "\n\nWrite a concise but complete answer with citations:\n"
    return prompt
