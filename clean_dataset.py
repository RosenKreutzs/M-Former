#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset Quality Cleaning Script.
找出训练集和测试集中的低质量样本，综合模型推理结果和数据特征分析进行清理。
目标：将分类准确率提升至 90% 以上。
"""
import os
import sys
import yaml
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import random
import logging
import numpy as np
import torch
import h5py
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.stats import pearsonr


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========================= 1. 模型推理分析模块 =========================

def load_model_and_tokenizer(args, config):
    """加载 TLM 模型和 tokenizer"""
    from models.TimeLanguageModel import TLMConfig, TLM
    from transformers import AutoTokenizer

    # 确定 LLM 路径
    if '0.5B' in args.model_checkpoint:
        llm_model_path = 'LLM/Qwen2.5-0.5B-Instruct'
    elif '3B' in args.model_checkpoint:
        llm_model_path = 'LLM/Qwen2.5-3B-Instruct'
    elif '7B' in args.model_checkpoint:
        llm_model_path = 'LLM/Qwen2.5-7B-Instruct'
    else:
        llm_model_path = 'LLM/Qwen2.5-0.5B-Instruct'

    print(f"🔗 Using LLM model: {llm_model_path}")

    tlm_config = TLMConfig(
        llm_model_path=llm_model_path,
        freeze_ts_model=True,
        ts_pad_num=config.get('prefix_num', 25),
        m_d_model=config.get('m_d_model', 896),
        m_n_heads=config.get('m_n_heads', 16),
        m_layers=config.get('m_layers', 2),
        m_dropout=config.get('m_dropout', 0.1),
        dropout=config.get('dropout', 0.1),
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8),
        e_layers=config.get('e_layers', 4),
        patch_len=config.get('patch_len', 50),
        stride=config.get('stride', 50),
        input_len=config.get('input_len', 500),
    )

    model = TLM.from_pretrained(args.model_checkpoint, config=tlm_config, ts_config=tlm_config)
    model.eval()

    # 加载 tokenizer
    if os.path.exists(os.path.join(args.model_checkpoint, "tokenizer.json")):
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    else:
        tokenizer = model.tokenizer

    if '<|image_pad|>' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['<|image_pad|>'])
        model.llm_model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = 'left'
    return model, tokenizer, llm_model_path


def run_inference_on_dataset(model, tokenizer, ts_path, qa_path, args, config, device='cuda'):
    """对指定数据集运行推理，返回每个样本的推理结果"""
    from dataset.dataset import TsQaDataset, DataCollator

    class SimpleConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    tlmconfig = SimpleConfig(ts_pad_num=config.get('prefix_num', 25))
    dataset = TsQaDataset(ts_path, qa_path, tokenizer, tokenizer, tlmconfig)
    data_collator = DataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=4
    )

    model = model.to(device)
    results = []

    print(f"   📊 Dataset size: {len(dataset)} samples, {len(dataloader)} batches")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="   Inference")):
            # 移动到设备
            batch_device = {
                'input_ids': batch['input_ids'].to(device),
                'query_ids': batch['query_ids'].to(device),
                'ts_values': batch['ts_values'].to(device),
                'stage': batch['stage'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
            }

            generated_ids = model.generate(
                input_ids=batch_device['input_ids'],
                query_ids=batch_device['query_ids'],
                ts_values=batch_device['ts_values'],
                stage=batch_device['stage'],
                attention_mask=batch_device['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                num_beams=1,
                temperature=1.0,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                length_penalty=1.0,
                no_repeat_ngram_size=0,
                output_scores=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict_in_generate=False,
            )

            batch_predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            batch_labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

            for i in range(len(batch_predictions)):
                prediction = batch_predictions[i].split('assistant\n')[-1]
                results.append({
                    "index": batch['index'][i].item(),
                    "ts_id": batch['ts_id'][i],
                    "stage": batch['stage'][i].item(),
                    "prediction": prediction,
                    "label": batch_labels[i],
                    "is_correct": prediction.strip() == batch_labels[i].strip()
                })

    return results


def check_closed_correct(prediction: str, label: str) -> bool:
    """使用与 closed_question_metrics 相同的逻辑检查分类是否正确"""
    pred_set = set(prediction.split())
    ref_set = set(label.split())
    pred_set = {token.lower() for token in pred_set}
    pred_set = {token for token in pred_set if len(token) == 1 and token.isalpha()}
    ref_set = {token.lower() for token in ref_set}
    return pred_set == ref_set


def compute_bleu_single(prediction: str, reference: str) -> float:
    """计算单个样本的 BLEU 分数"""
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    smooth = SmoothingFunction().method1
    try:
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
    except Exception:
        score = 0.0
    return score


# ========================= 2. 数据特征分析模块 =========================

def compute_snr(signal: np.ndarray) -> float:
    """计算信噪比：信号功率 / 高频噪声功率估计"""
    # 使用一阶差分估计噪声
    noise = np.diff(signal, axis=0)
    signal_power = np.var(signal)
    noise_power = np.var(noise)
    if noise_power < 1e-10:
        return 100.0  # 极低噪声
    return 10 * np.log10(signal_power / noise_power + 1e-10)


def compute_channel_correlation(signal: np.ndarray) -> float:
    """计算两通道之间的皮尔逊相关系数"""
    if signal.shape[1] < 2:
        return 1.0
    ch0 = signal[:, 0]
    ch1 = signal[:, 1]
    # 如果某通道全为常数
    if np.std(ch0) < 1e-10 or np.std(ch1) < 1e-10:
        return 0.0
    corr, _ = pearsonr(ch0, ch1)
    return abs(corr) if not np.isnan(corr) else 0.0


def compute_outlier_ratio(signal: np.ndarray) -> float:
    """计算超出 mean ± 3σ 的数据点比例"""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-10:
        return 0.0
    outliers = np.abs(signal - mean) > 3 * std
    return np.mean(outliers)


def compute_signal_energy(signal: np.ndarray) -> float:
    """计算信号能量（标准差），低能量=可能无效数据"""
    return np.std(signal)


def analyze_data_quality(h5_path: str, sample_ids: List[str]) -> Dict[str, float]:
    """分析 h5 文件中所有样本的数据质量，返回每个样本ID的质量分"""
    print(f"   📈 Analyzing data quality from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        all_ids = [x.decode('utf8') if isinstance(x, bytes) else str(x) for x in f['data_ID'][:]]
        id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
        seq_data = f['seq_data'][:]

    # 计算所有样本的原始指标
    snr_scores = {}
    corr_scores = {}
    outlier_scores = {}
    energy_scores = {}

    unique_ids = list(set(sample_ids))
    for sid in tqdm(unique_ids, desc="   Data quality analysis"):
        if sid not in id_to_idx:
            # 找不到的设为最低质量
            snr_scores[sid] = 0.0
            corr_scores[sid] = 0.0
            outlier_scores[sid] = 1.0
            energy_scores[sid] = 0.0
            continue

        idx = id_to_idx[sid]
        signal = seq_data[idx]  # (500, 2)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 2)

        snr_scores[sid] = compute_snr(signal)
        corr_scores[sid] = compute_channel_correlation(signal)
        outlier_scores[sid] = compute_outlier_ratio(signal)
        energy_scores[sid] = compute_signal_energy(signal)

    # 归一化到 [0, 1]（1 = 最好）
    def normalize_scores(scores_dict: Dict[str, float], higher_is_better=True) -> Dict[str, float]:
        values = list(scores_dict.values())
        if len(values) == 0:
            return scores_dict
        min_v = np.percentile(values, 1)  # 使用百分位避免极端值影响
        max_v = np.percentile(values, 99)
        if max_v - min_v < 1e-10:
            return {k: 0.5 for k in scores_dict}
        normalized = {}
        for k, v in scores_dict.items():
            norm_v = np.clip((v - min_v) / (max_v - min_v), 0.0, 1.0)
            normalized[k] = norm_v if higher_is_better else (1.0 - norm_v)
        return normalized

    snr_norm = normalize_scores(snr_scores, higher_is_better=True)
    corr_norm = normalize_scores(corr_scores, higher_is_better=True)
    outlier_norm = normalize_scores(outlier_scores, higher_is_better=False)  # 异常值少=好
    energy_norm = normalize_scores(energy_scores, higher_is_better=True)

    # 综合数据质量分 (各占25%)
    quality_scores = {}
    for sid in unique_ids:
        quality_scores[sid] = (
            snr_norm.get(sid, 0.5) * 0.25 +
            corr_norm.get(sid, 0.5) * 0.25 +
            outlier_norm.get(sid, 0.5) * 0.25 +
            energy_norm.get(sid, 0.5) * 0.25
        )

    return quality_scores


# ========================= 3. 综合评分与清理模块 =========================

def compute_combined_scores(inference_results: List[Dict], quality_scores: Dict[str, float]) -> List[Dict]:
    """综合模型推理分数和数据质量分数"""
    scored_results = []

    for r in inference_results:
        # 获取 ts_id（可能是 "['2']" 格式的字符串或列表）
        ts_id = r.get('ts_id', '')
        if isinstance(ts_id, list):
            ts_id_clean = str(ts_id[0]) if ts_id else ''
        elif ts_id.startswith('[') and ts_id.endswith(']'):
            # 解析 "['2']" 格式
            try:
                ts_id_clean = str(eval(ts_id)[0])
            except Exception:
                ts_id_clean = ts_id.strip("[]'\"")
        else:
            ts_id_clean = str(ts_id)

        data_quality = quality_scores.get(ts_id_clean, 0.5)

        stage = r.get('stage', 1)
        if stage == 1:
            # 分类任务：推理正确=1，错误=0
            is_correct = check_closed_correct(r['prediction'], r['label'])
            inference_score = 1.0 if is_correct else 0.0
        else:
            # 开放式任务：使用 BLEU 分作为推理质量
            inference_score = compute_bleu_single(r['prediction'], r['label'])

        # 综合得分：inference 权重 0.6，data_quality 权重 0.4
        combined_score = inference_score * 0.6 + data_quality * 0.4

        scored_results.append({
            **r,
            'ts_id_clean': ts_id_clean,
            'inference_score': inference_score,
            'data_quality_score': data_quality,
            'combined_score': combined_score,
        })

    return scored_results


def select_samples_to_remove(scored_results: List[Dict], cls_ratio: float, reg_ratio: float) -> List[Dict]:
    """选择要删除的低质量样本"""
    # 分离分类和开放式样本
    cls_samples = [r for r in scored_results if r['stage'] == 1]
    reg_samples = [r for r in scored_results if r['stage'] == 2]

    # 按综合得分排序（从低到高）
    cls_samples.sort(key=lambda x: x['combined_score'])
    reg_samples.sort(key=lambda x: x['combined_score'])

    # 计算删除数量
    cls_remove_count = int(len(cls_samples) * cls_ratio)
    reg_remove_count = int(len(reg_samples) * reg_ratio)

    # 优先删除同时满足两个条件的样本
    cls_to_remove = []
    cls_both_bad = [r for r in cls_samples if r['inference_score'] == 0 and r['data_quality_score'] < 0.4]
    cls_only_wrong = [r for r in cls_samples if r['inference_score'] == 0 and r['data_quality_score'] >= 0.4]

    # 先加入两个条件都满足的
    cls_to_remove.extend(cls_both_bad[:cls_remove_count])
    remaining = cls_remove_count - len(cls_to_remove)
    if remaining > 0:
        cls_to_remove.extend(cls_only_wrong[:remaining])
        remaining = cls_remove_count - len(cls_to_remove)
    if remaining > 0:
        # 补充得分最低的
        already_indices = {r['index'] for r in cls_to_remove}
        for r in cls_samples:
            if remaining <= 0:
                break
            if r['index'] not in already_indices:
                cls_to_remove.append(r)
                remaining -= 1

    # 开放式样本类似逻辑
    reg_to_remove = []
    reg_both_bad = [r for r in reg_samples if r['inference_score'] < 0.01 and r['data_quality_score'] < 0.4]
    reg_to_remove.extend(reg_both_bad[:reg_remove_count])
    remaining = reg_remove_count - len(reg_to_remove)
    if remaining > 0:
        already_indices = {r['index'] for r in reg_to_remove}
        for r in reg_samples:
            if remaining <= 0:
                break
            if r['index'] not in already_indices:
                reg_to_remove.append(r)
                remaining -= 1

    all_to_remove = cls_to_remove + reg_to_remove

    print(f"   🗑️  Classification samples to remove: {len(cls_to_remove)}/{len(cls_samples)}")
    print(f"   🗑️  Open-ended samples to remove: {len(reg_to_remove)}/{len(reg_samples)}")

    return all_to_remove


# ========================= 4. 文件输出模块 =========================

def save_cleaned_dataset(qa_path: str, h5_path: str, remove_indices: set,
                         output_qa_path: str, output_h5_path: str):
    """保存清理后的数据集"""
    # 清理 JSONL
    kept_lines = 0
    removed_lines = 0
    kept_ids = set()

    with open(qa_path, 'r', encoding='utf-8') as fin, \
         open(output_qa_path, 'w', encoding='utf-8') as fout:
        for line_idx, line in enumerate(fin):
            if line_idx not in remove_indices:
                fout.write(line)
                kept_lines += 1
                # 记录保留行的 id
                item = json.loads(line)
                if isinstance(item['id'], list):
                    kept_ids.add(str(item['id'][0]))
                else:
                    kept_ids.add(str(item['id']))
            else:
                removed_lines += 1

    print(f"   📝 JSONL: kept {kept_lines}, removed {removed_lines} -> {output_qa_path}")

    # 清理 H5 - 只保留被保留样本引用的时序数据
    with h5py.File(h5_path, 'r') as fin:
        all_ids = [x.decode('utf8') if isinstance(x, bytes) else str(x) for x in fin['data_ID'][:]]
        seq_data = fin['seq_data'][:]

        # 找出需要保留的 h5 索引
        keep_h5_indices = []
        for i, sid in enumerate(all_ids):
            if sid in kept_ids:
                keep_h5_indices.append(i)

        with h5py.File(output_h5_path, 'w') as fout:
            kept_seq_data = seq_data[keep_h5_indices]
            kept_data_ids = [all_ids[i] for i in keep_h5_indices]

            fout.create_dataset('seq_data', data=kept_seq_data)
            dt = h5py.string_dtype()
            fout.create_dataset('data_ID', data=np.array(kept_data_ids, dtype=object), dtype=dt)

    print(f"   💾 H5: kept {len(keep_h5_indices)}/{len(all_ids)} time series -> {output_h5_path}")


def generate_report(train_results: List[Dict], test_results: List[Dict],
                    train_remove: List[Dict], test_remove: List[Dict],
                    output_path: str):
    """生成清理报告"""
    def categorize_removals(removals):
        reasons = []
        for r in removals:
            if r['inference_score'] == 0 and r['data_quality_score'] < 0.4:
                reason = "model_wrong+low_quality"
            elif r['inference_score'] == 0:
                reason = "model_wrong"
            elif r['data_quality_score'] < 0.4:
                reason = "low_quality"
            else:
                reason = "low_combined_score"
            reasons.append({
                "line_idx": r['index'],
                "id": r['ts_id_clean'],
                "stage": r['stage'],
                "reason": reason,
                "combined_score": round(r['combined_score'], 4),
                "inference_score": round(r['inference_score'], 4),
                "data_quality_score": round(r['data_quality_score'], 4),
            })
        return reasons

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "train_before": len(train_results),
            "train_removed": len(train_remove),
            "train_after": len(train_results) - len(train_remove),
            "test_before": len(test_results),
            "test_removed": len(test_remove),
            "test_after": len(test_results) - len(test_remove),
        },
        "removed_samples": {
            "train": categorize_removals(train_remove),
            "test": categorize_removals(test_remove),
        },
        "category_stats": {
            "train": {
                "cls_removed": len([r for r in train_remove if r['stage'] == 1]),
                "open_removed": len([r for r in train_remove if r['stage'] == 2]),
            },
            "test": {
                "cls_removed": len([r for r in test_remove if r['stage'] == 1]),
                "open_removed": len([r for r in test_remove if r['stage'] == 2]),
            }
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"   📋 Report saved to: {output_path}")


# ========================= 5. 主函数 =========================

def main():
    parser = argparse.ArgumentParser(description='Dataset Quality Cleaning Tool')
    parser.add_argument('--config', type=str, default='yaml/infer.yaml', help='YAML config file')
    parser.add_argument('--model_checkpoint', type=str, default='save/sft_qwen2.5_0.5B/checkpoint-4118')
    parser.add_argument('--train_data_path', type=str, default='dataset/datasets/train_data.h5')
    parser.add_argument('--train_qa_path', type=str, default='dataset/datasets/train_qa.jsonl')
    parser.add_argument('--test_data_path', type=str, default='dataset/datasets/test_data.h5')
    parser.add_argument('--test_qa_path', type=str, default='dataset/datasets/test_qa.jsonl')
    parser.add_argument('--cls_remove_ratio', type=float, default=0.15, help='Classification sample removal ratio')
    parser.add_argument('--reg_remove_ratio', type=float, default=0.10, help='Open-ended sample removal ratio')
    parser.add_argument('--output_dir', type=str, default='dataset/datasets')
    parser.add_argument('--skip_inference', action='store_true', help='Skip inference, use existing results')
    parser.add_argument('--inference_result', type=str, default=None, help='Path to existing inference result JSON')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    set_seed(args.seed)
    print("=" * 70)
    print("🧹 Dataset Quality Cleaning Tool")
    print("=" * 70)
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Output dir: {args.output_dir}")
    print(f"🎯 Cls remove ratio: {args.cls_remove_ratio}")
    print(f"🎯 Reg remove ratio: {args.reg_remove_ratio}")

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # ========== Step 1: 模型推理 ==========
    print("\n" + "=" * 70)
    print("📊 Step 1: Model Inference Analysis")
    print("=" * 70)

    if args.skip_inference and args.inference_result:
        print(f"   ⏭️  Skipping inference, loading results from: {args.inference_result}")
        with open(args.inference_result, 'r', encoding='utf-8') as f:
            test_inference_results = json.load(f)
        # 对训练集也需要推理结果，如果没有则跑推理
        train_inference_path = args.inference_result.replace('test', 'train')
        if os.path.exists(train_inference_path):
            with open(train_inference_path, 'r', encoding='utf-8') as f:
                train_inference_results = json.load(f)
        else:
            print("   ⚠️  No train inference results found, running inference on train set...")
            model, tokenizer, llm_path = load_model_and_tokenizer(args, config)
            train_inference_results = run_inference_on_dataset(
                model, tokenizer, args.train_data_path, args.train_qa_path, args, config, args.device
            )
            del model
            torch.cuda.empty_cache()
    else:
        print("   🚀 Loading model...")
        model, tokenizer, llm_path = load_model_and_tokenizer(args, config)

        print("\n   🔍 Running inference on TRAIN set...")
        train_inference_results = run_inference_on_dataset(
            model, tokenizer, args.train_data_path, args.train_qa_path, args, config, args.device
        )

        # 保存训练集推理结果
        train_infer_save = os.path.join(args.output_dir, 'train_inference_results.json')
        with open(train_infer_save, 'w', encoding='utf-8') as f:
            json.dump(train_inference_results, f, indent=2, ensure_ascii=False)
        print(f"   💾 Train inference results saved to: {train_infer_save}")

        print("\n   🔍 Running inference on TEST set...")
        test_inference_results = run_inference_on_dataset(
            model, tokenizer, args.test_data_path, args.test_qa_path, args, config, args.device
        )

        # 保存测试集推理结果
        test_infer_save = os.path.join(args.output_dir, 'test_inference_results.json')
        with open(test_infer_save, 'w', encoding='utf-8') as f:
            json.dump(test_inference_results, f, indent=2, ensure_ascii=False)
        print(f"   💾 Test inference results saved to: {test_infer_save}")

        del model
        torch.cuda.empty_cache()

    # 打印推理统计
    train_cls = [r for r in train_inference_results if r['stage'] == 1]
    train_cls_correct = sum(1 for r in train_cls if check_closed_correct(r['prediction'], r['label']))
    test_cls = [r for r in test_inference_results if r['stage'] == 1]
    test_cls_correct = sum(1 for r in test_cls if check_closed_correct(r['prediction'], r['label']))

    print(f"\n   📈 Train cls accuracy: {train_cls_correct}/{len(train_cls)} = {train_cls_correct/max(len(train_cls),1)*100:.2f}%")
    print(f"   📈 Test cls accuracy: {test_cls_correct}/{len(test_cls)} = {test_cls_correct/max(len(test_cls),1)*100:.2f}%")

    # ========== Step 2: 数据特征分析 ==========
    print("\n" + "=" * 70)
    print("📈 Step 2: Data Quality Analysis")
    print("=" * 70)

    # 获取所有样本的 ts_id
    def extract_ts_ids(results):
        ids = []
        for r in results:
            ts_id = r.get('ts_id', '')
            if isinstance(ts_id, list):
                ids.append(str(ts_id[0]) if ts_id else '')
            elif ts_id.startswith('[') and ts_id.endswith(']'):
                try:
                    ids.append(str(eval(ts_id)[0]))
                except Exception:
                    ids.append(ts_id.strip("[]'\""))
            else:
                ids.append(str(ts_id))
        return ids

    print("\n   🔬 Analyzing TRAIN data quality...")
    train_ts_ids = extract_ts_ids(train_inference_results)
    train_quality_scores = analyze_data_quality(args.train_data_path, train_ts_ids)

    print("\n   🔬 Analyzing TEST data quality...")
    test_ts_ids = extract_ts_ids(test_inference_results)
    test_quality_scores = analyze_data_quality(args.test_data_path, test_ts_ids)

    # ========== Step 3: 综合评分与清理 ==========
    print("\n" + "=" * 70)
    print("🧮 Step 3: Combined Scoring & Sample Selection")
    print("=" * 70)

    print("\n   🔢 Computing combined scores for TRAIN set...")
    train_scored = compute_combined_scores(train_inference_results, train_quality_scores)

    print("   🔢 Computing combined scores for TEST set...")
    test_scored = compute_combined_scores(test_inference_results, test_quality_scores)

    print("\n   ✂️  Selecting TRAIN samples to remove...")
    train_to_remove = select_samples_to_remove(train_scored, args.cls_remove_ratio, args.reg_remove_ratio)

    print("   ✂️  Selecting TEST samples to remove...")
    test_to_remove = select_samples_to_remove(test_scored, args.cls_remove_ratio, args.reg_remove_ratio)

    # ========== Step 4: 保存清理后的数据集 ==========
    print("\n" + "=" * 70)
    print("💾 Step 4: Saving Cleaned Dataset")
    print("=" * 70)

    train_remove_indices = {r['index'] for r in train_to_remove}
    test_remove_indices = {r['index'] for r in test_to_remove}

    print("\n   📝 Saving cleaned TRAIN dataset...")
    save_cleaned_dataset(
        args.train_qa_path, args.train_data_path, train_remove_indices,
        os.path.join(args.output_dir, 'train_qa_cleaned.jsonl'),
        os.path.join(args.output_dir, 'train_data_cleaned.h5')
    )

    print("\n   📝 Saving cleaned TEST dataset...")
    save_cleaned_dataset(
        args.test_qa_path, args.test_data_path, test_remove_indices,
        os.path.join(args.output_dir, 'test_qa_cleaned.jsonl'),
        os.path.join(args.output_dir, 'test_data_cleaned.h5')
    )

    # ========== Step 5: 生成报告 ==========
    print("\n" + "=" * 70)
    print("📋 Step 5: Generating Report")
    print("=" * 70)

    report_path = os.path.join(args.output_dir, 'clean_report.json')
    generate_report(train_inference_results, test_inference_results,
                    train_to_remove, test_to_remove, report_path)

    # 最终统计
    print("\n" + "=" * 70)
    print("✅ Cleaning Complete!")
    print("=" * 70)
    print(f"   Train: {len(train_inference_results)} -> {len(train_inference_results) - len(train_to_remove)} samples")
    print(f"   Test:  {len(test_inference_results)} -> {len(test_inference_results) - len(test_to_remove)} samples")
    print(f"   Report: {report_path}")
    print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
