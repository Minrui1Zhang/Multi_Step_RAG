from __future__ import annotations

import json
from pathlib import Path

from .config import dump_yaml
from .evaluation import evaluate_example, reduce_metrics
from .extraction import run_extraction
from .generation import OfflineGenerator, build_prompt
from .reflection import EntailmentScorer, filter_sentences, reflect_answer
from .retrieval import rerank
from .utils import load_jsonl, set_seed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def _model_path(model_paths: dict, config: dict, field_name: str) -> str:
    override = config["generation"].get("override_model_path", "")
    if override:
        return override
    key = config[field_name].get("verifier_model_key") if field_name != "generation" else config["generation"].get("model_key", "generator_model")
    return model_paths[key]


def run_pipeline(config: dict, model_paths: dict) -> dict:
    set_seed(int(config["runtime"].get("random_seed", 42)))
    rows = load_jsonl(config["dataset_path"])
    limit = int(config["runtime"].get("limit", 0))
    if limit > 0:
        rows = rows[:limit]

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(config, output_dir / "config.resolved.yaml")

    generator = None
    if config["generation"].get("use_llm", True):
        generator = OfflineGenerator(
            model_path=_model_path(model_paths, config, "generation"),
            dtype=model_paths.get("generator_dtype", "auto"),
            device=model_paths.get("device", "cuda"),
            max_new_tokens=int(config["generation"].get("max_new_tokens", model_paths.get("max_new_tokens", 384))),
            temperature=float(config["generation"].get("temperature", 0.0)),
        )

    filter_scorer = None
    if config["filter"].get("mode") == "nli":
        filter_scorer = EntailmentScorer(
            model_path=model_paths[config["evaluation"].get("nli_model_key", "nli_model")],
            dtype=model_paths.get("nli_dtype", "auto"),
            device=model_paths.get("device", "cuda"),
        )

    reflection_scorer = None
    if config["reflection"].get("mode") in {"nli", "attrscore"}:
        reflection_scorer = EntailmentScorer(
            model_path=model_paths[config["reflection"].get("verifier_model_key", "nli_model")],
            dtype=model_paths.get("nli_dtype", "auto"),
            device=model_paths.get("device", "cuda"),
        )

    eval_scorer = None
    if config["evaluation"].get("citation", True) or config["evaluation"].get("claims_nli", True):
        eval_scorer = EntailmentScorer(
            model_path=model_paths[config["evaluation"].get("nli_model_key", "nli_model")],
            dtype=model_paths.get("nli_dtype", "auto"),
            device=model_paths.get("device", "cuda"),
        )

    metric_rows = []
    prediction_file = output_dir / "predictions.jsonl"

    with prediction_file.open("w", encoding="utf-8") as f:
        for example in tqdm(rows, desc=config["experiment_name"]):
            retrieval_output = rerank(example, config["retrieval"])
            retrieval_output = filter_sentences(example["question"], retrieval_output, filter_scorer, config["filter"])
            extracted = run_extraction(retrieval_output, config["extract"])
            prompt = build_prompt(example, retrieval_output, extracted)

            if generator is None:
                prediction = extracted[0]["fact"] + " " + extracted[0]["citation"] if extracted else "Insufficient evidence."
            else:
                prediction = generator.generate(prompt)

            reflection_result = reflect_answer(prediction, retrieval_output, extracted, reflection_scorer, config["reflection"])
            reflected_answer = reflection_result["answer"]

            metrics = evaluate_example(
                example,
                reflected_answer,
                retrieval_output["docs"],
                eval_scorer,
                config["evaluation"],
            )
            metric_rows.append(metrics)

            row = {
                "id": example["id"],
                "dataset": example["dataset"],
                "question": example["question"],
                "target": example.get("target", ""),
                "prediction": prediction,
                "reflected_answer": reflected_answer,
                "retrieved_docs": retrieval_output["docs"],
                "retrieved_sentences": retrieval_output["sentences"],
                "extracted": extracted,
                "reflection_trace": reflection_result["reflection_trace"],
                "metrics": metrics,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = reduce_metrics(metric_rows)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
