python scripts/run_weak_sft_experiment.py \
  --predictions outputs/run400/exp07_reflect_nli_asqa_eval_top20/predictions.jsonl \
  --sft_output data/sft/asqa_train_reflection.jsonl \
  --max_samples 1500 \
  --train_config configs/lora_sft.yaml \
  --models configs/model_paths.yaml \
  --base_model_key generator_model \
  --adapter_output_dir outputs/sft/llama_asqa_lora \
  --merged_output_dir outputs/sft/llama_asqa_lora_merged \
  --merged_model_key llama_asqa_lora_model \
  --eval_base_config configs/base_experiment.yaml \
  --eval_datasets data/processed/asqa_eval_top20.jsonl \
  --eval_output_root outputs/runs/weak_sft_asqa 


python scripts/run_weak_sft_experiment.py \
  --predictions outputs/run400/exp07_reflect_nli_eli5_eval_top20/predictions.jsonl \
  --sft_output data/sft/eli5_train_reflection.jsonl \
  --max_samples 1500 \
  --train_config configs/lora_sft.yaml \
  --models configs/model_paths.yaml \
  --base_model_key generator_model \
  --adapter_output_dir outputs/sft/llama_eli5_lora \
  --merged_output_dir outputs/sft/llama_eli5_lora_merged \
  --merged_model_key llama_eli5_lora_model \
  --eval_base_config configs/base_experiment.yaml \
  --eval_datasets data/processed/eli5_eval_top20.jsonl \
  --eval_output_root outputs/runs/weak_sft_eli5

python scripts/run_weak_sft_experiment.py \
  --predictions outputs/run400/exp07_reflect_nli_qampari_eval_top20/predictions.jsonl \
  --sft_output data/sft/qampari_train_reflection.jsonl \
  --max_samples 1500 \
  --train_config configs/lora_sft.yaml \
  --models configs/model_paths.yaml \
  --base_model_key generator_model \
  --adapter_output_dir outputs/sft/llama_qampari_lora \
  --merged_output_dir outputs/sft/llama_qampari_lora_merged \
  --merged_model_key llama_qampari_lora_model \
  --eval_base_config configs/base_experiment.yaml \
  --eval_datasets data/processed/qampari_eval_top20.jsonl \
  --eval_output_root outputs/runs/weak_sft_qampari  
