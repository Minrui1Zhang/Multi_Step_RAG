#!/bin/bash
# ==========================================
# 单组实验测试脚本 (Smoke Test)
# ==========================================

DATASET="data/processed/asqa_eval_top20.jsonl"
CONFIG="configs/base_experiment.yaml"
MODELS="configs/model_paths.yaml"
OUTPUT="outputs/runs/asqa_smoke"

echo ">>> 运行 Smoke Test (100 条样本)"
python scripts/run_pipeline.py \
    --config "$CONFIG" \
    --models "$MODELS" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --limit 100

if [ $? -eq 0 ]; then
    echo "✓ Smoke Test 完成"
    echo ">>> 输出: $OUTPUT/predictions.jsonl"
else
    echo "✗ Smoke Test 失败"
    exit 1
fi