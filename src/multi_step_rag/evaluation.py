from __future__ import annotations

from rouge_score import rouge_scorer

from .reflection import EntailmentScorer
from .utils import extract_citation_ids, normalize_answer, remove_citations, split_sentences


def compute_rouge(reference: str, prediction: str) -> float:
    """计算 ROUGE-L Sum 分数 (文本相似度/流畅度参考)"""
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    ref = "\n".join(split_sentences(reference.lower()))
    pred = "\n".join(split_sentences(prediction.lower()))
    return 100.0 * scorer.score(ref, pred)["rougeLsum"].fmeasure


def exact_presence(short_answers: list[str], context: str) -> bool:
    """检查短答案是否出现在上下文中"""
    n_context = normalize_answer(context)
    for ans in short_answers:
        if normalize_answer(ans) in n_context:
            return True
    return False


# ==========================================
# ASQA 评估指标 (参考 ALCE 论文)
# ==========================================
def evaluate_asqa(example: dict, prediction: str) -> dict:
    """
    ASQA 数据集评估:
    - 正确性: Exact Match Recall (EM 召回率) - 检查是否包含标准答案短文本
    - 引用质量: Citation Recall / Precision (继承自 evaluate_citations)
    """
    qa_pairs = example.get("qa_pairs") or []
    if not qa_pairs:
        return {}
    
    scores = []
    for pair in qa_pairs:
        scores.append(float(exact_presence(pair["short_answers"], prediction)))
    
    # EM 召回率: 包含的短答案数 / 总短答案数
    em_recall = 100.0 * (sum(scores) / len(scores))
    # str_hit: 所有短答案都出现才为 100
    str_hit = 100.0 * float(all(score == 1.0 for score in scores))
    
    return {
        "em_recall": em_recall,      # ALCE: Exact Match Recall
        "str_hit": str_hit,          # 全部匹配标志
    }


# ==========================================
# QAMPARI 评估指标 (参考 ALCE 论文)
# ==========================================
def evaluate_qampari(example: dict, prediction: str) -> dict:
    """
    QAMPARI 数据集评估:
    - 正确性: Precision、Recall-5
      - Precision: 实体完全匹配精确率
      - Recall-5: 若预测包含至少 5 个正确答案，直接视为召回率 100%
    - 引用质量: Citation Recall / Precision
    """
    gold_groups = example.get("answers") or []  # [[ans1, ans2], [ans3], ...]
    if not gold_groups:
        return {}
    
    pred_text = normalize_answer(remove_citations(prediction))
    
    # 计算命中数 (每个 group 只需命中一个即可)
    hits = 0
    for group in gold_groups:
        group_hit = any(normalize_answer(candidate) in pred_text for candidate in group)
        hits += int(group_hit)
    
    # 解析预测的实体列表
    pred_units = [unit.strip() for unit in remove_citations(prediction).replace("\n", ",").split(",") if unit.strip()]
    
    # Precision: 命中的 group 数 / 预测的实体数
    precision = hits / max(len(pred_units), 1)
    
    # Recall: 命中的 group 数 / 总 group 数
    recall = hits / len(gold_groups)
    
    # Recall-5: 若预测包含至少 5 个正确答案，直接视为召回率 100%
    recall_5 = 100.0 if hits >= 5 else 100.0 * recall
    
    # F1
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    
    return {
        "list_precision": 100.0 * precision,    # 实体匹配精确率
        "list_recall": 100.0 * recall,          # 实体匹配召回率
        "list_recall_5": recall_5,              # Recall-5 (ALCE 特色)
        "list_f1": 100.0 * f1,
    }


# ==========================================
# ELI5 评估指标 (参考 ALCE 论文)
# ==========================================
def evaluate_eli5(example: dict, prediction: str, scorer: EntailmentScorer | None) -> dict:
    """
    ELI5 数据集评估:
    - 正确性: Claim Recall
      - 用 NLI 模型判断生成内容是否蕴含标准答案中的子论断
    - 引用质量: Citation Recall / Precision
    - 流畅度: MAUVE (可选，需额外依赖)
    """
    # ELI5 数据集没有 qa_pairs，使用 claims 字段
    claims = example.get("claims") or []
    if not claims:
        return {}
    
    premise = remove_citations(prediction)
    values = []
    for claim in claims:
        if scorer is None:
            values.append(0.0)
        else:
            # NLI 验证: 预测是否蕴含论断
            values.append(float(scorer.score(premise, claim) >= 0.5))
    
    # Claim Recall: 被蕴含的论断数 / 总论断数
    claim_recall = 100.0 * sum(values) / len(values)
    
    return {
        "claim_recall": claim_recall,  # ALCE: Claim Recall
    }


def evaluate_claims(example: dict, prediction: str, scorer: EntailmentScorer | None) -> dict:
    """兼容旧接口，用于非 ELI5 数据集的 claims 评估"""
    claims = example.get("claims") or []
    if not claims:
        return {}
    premise = remove_citations(prediction)
    values = []
    for claim in claims:
        if scorer is None:
            values.append(0.0)
        else:
            values.append(float(scorer.score(premise, claim) >= 0.5))
    return {
        "claims_nli": 100.0 * sum(values) / len(values),
    }


def evaluate_citations(prediction: str, docs: list[dict], scorer: EntailmentScorer | None, at_most_citations: int = 3) -> dict:
    """
    引用质量评估 (ALCE 风格):
    - Citation Recall: 预测句被引用文档支持的比例
    - Citation Precision: 引用正确的比例
    """
    # This follows the sentence-level citation support idea used in ALCE/AutoAIS,
    # but is simplified for offline local evaluation on a single server.
    doc_map = {doc["doc_id"]: doc for doc in docs}
    pred_sents = split_sentences(prediction)
    if not pred_sents:
        return {
            "citation_rec": 0.0,
            "citation_prec": 0.0,
        }

    sent_recalls = []
    cite_precisions = []

    for sent in pred_sents:
        citation_ids = extract_citation_ids(sent)[:at_most_citations]
        clean_sent = remove_citations(sent)
        if not citation_ids:
            sent_recalls.append(0.0)
            continue

        joint_passage = " ".join(doc_map[cid]["text"] for cid in citation_ids if cid in doc_map)
        supported = scorer.score(joint_passage, clean_sent) >= 0.5 if scorer is not None else False
        sent_recalls.append(float(supported))

        correct = 0
        for cid in citation_ids:
            if cid not in doc_map:
                continue
            if scorer is not None and scorer.score(doc_map[cid]["text"], clean_sent) >= 0.5:
                correct += 1
        cite_precisions.append(correct / len(citation_ids))

    return {
        "citation_rec": 100.0 * sum(sent_recalls) / len(sent_recalls),
        "citation_prec": 100.0 * sum(cite_precisions) / len(cite_precisions) if cite_precisions else 0.0,
    }


def evaluate_example(example: dict, prediction: str, docs: list[dict], scorer: EntailmentScorer | None, cfg: dict) -> dict:
    """
    综合评估函数 - 根据数据集类型选择对应指标
    
    评估维度:
    1. 流畅度: rougeLsum (MAUVE 需要额外依赖，可后续添加)
    2. 正确性: 
       - ASQA: em_recall (Exact Match Recall)
       - QAMPARI: list_precision, list_recall, list_recall_5
       - ELI5: claim_recall
    3. 引用质量: citation_rec, citation_prec
    """
    metrics = {}
    target = example.get("target", "")
    
    # 1. 流畅度 (参考指标)
    if cfg.get("rouge", True) and target:
        metrics["rougeLsum"] = compute_rouge(target, remove_citations(prediction))
    
    # 2. 正确性指标 (按数据集)
    dataset = example.get("dataset", "")
    if dataset == "asqa":
        # ASQA: Exact Match Recall
        metrics.update(evaluate_asqa(example, prediction))
    elif dataset == "qampari":
        # QAMPARI: Precision, Recall, Recall-5
        metrics.update(evaluate_qampari(example, prediction))
    elif dataset == "eli5":
        # ELI5: Claim Recall (需要 NLI scorer)
        if cfg.get("claims_nli", True):
            metrics.update(evaluate_eli5(example, prediction, scorer))
    elif cfg.get("claims_nli", True):
        # 兼容旧接口
        metrics.update(evaluate_claims(example, prediction, scorer))
    
    # 3. 引用质量
    if cfg.get("citation", True):
        metrics.update(evaluate_citations(prediction, docs, scorer, int(cfg.get("at_most_citations", 3))))
    
    return metrics


def reduce_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    summary = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        summary[key] = sum(values) / len(values)
    return summary
