#!/bin/bash

# Dataset Quality Cleaning Script
# 使用模型推理 + 数据特征分析综合清理低质量样本

# 忽略 TRANSFORMERS_CACHE 废弃警告
export PYTHONWARNINGS="ignore::FutureWarning:transformers.utils.hub"

echo "=========================================="
echo "🧹 Starting Dataset Quality Cleaning..."
echo "=========================================="

accelerate launch --config_file yaml/accelerate_config.yaml clean_dataset.py \
    --config yaml/infer.yaml \
    --model_checkpoint save/sft_qwen2.5_0.5B/checkpoint-1800 \
    --train_data_path dataset/datasets/train_data.h5 \
    --train_qa_path dataset/datasets/train_qa.jsonl \
    --test_data_path dataset/datasets/test_data.h5 \
    --test_qa_path dataset/datasets/test_qa.jsonl \
    --cls_remove_ratio 0.35 \
    --reg_remove_ratio 0.25 \
    --output_dir dataset/datasets \
    --batch_size 256 \
    --max_new_tokens 128 \
    2>&1 | tee clean_dataset.log

echo ""
echo "✅ Cleaning finished. Log saved to: clean_dataset.log"
echo "📋 Report: dataset/datasets/clean_report.json"
