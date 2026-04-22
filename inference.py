#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference script for Time Language Model (TLM).
Performs inference on the test set and saves results and evaluation metrics.
"""
import os
import yaml
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*TRANSFORMERS_CACHE.*")
import argparse
import random
import logging
from transformers.utils import logging as transformers_logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoProcessor
from dataset.dataset import TsQaDataset, DataCollator
from models.TimeLanguageModel import TLM
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.metrics import open_question_metrics, closed_question_metrics
from typing import List, Dict, Any
from accelerate import Accelerator  

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_results(results, output_dir, config_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"inference_results_{config_name}_{timestamp}.json"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Inference results saved to: {filepath}")

def count_model_parameters(model):
    def count_params(module):
        if module is None:
            return 0
        return sum(p.numel() for p in module.parameters())
    llm_params = count_params(model.llm_model) / 1e6
    mformer_params = count_params(model.mformer) / 1e6
    ts_encoder_params = count_params(model.ts_encoder) / 1e6
    total_params = count_params(model) / 1e6
    return {
        'llm': llm_params,
        'mformer': mformer_params,
        'ts_encoder': ts_encoder_params,
        'total': total_params
    }
from accelerate import InitProcessGroupKwargs
from datetime import timedelta
def main_inference(args):
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
    
    # 设置日志级别，主进程显示进度，从进程静默
    if accelerator.is_local_main_process:
        transformers_logging.set_verbosity_info()
    else:
        transformers_logging.set_verbosity_error()
        logging.disable(logging.CRITICAL)

    set_seed(args.seed)
    if accelerator.is_main_process:
        print("🚀 Starting inference process...")
        print(f"🔧 Using config file: {args.config}")
        print(f"💾 Results will be saved to: {args.output_dir}")
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    args.__dict__.update(config)
    if accelerator.is_main_process:
        print("\n⚙️ Loading model from checkpoint...")
    # Import TLMConfig
    from models.TimeLanguageModel import TLMConfig
    
    # Determine LLM model path based on mformer model size
    if '0.5B' in args.model_checkpoint:
        llm_model_path = 'LLM/Qwen2.5-0.5B-Instruct'
    elif '3B' in args.model_checkpoint:
        llm_model_path = 'LLM/Qwen2.5-3B-Instruct'
    elif '7B' in args.model_checkpoint:
        llm_model_path = 'LLM/Qwen2.5-7B-Instruct'
    else:
        llm_model_path = 'LLM/Qwen2.5-7B-Instruct'

    args.llm_model_path = llm_model_path
    
    if accelerator.is_main_process:
        print(f"🔗 Using LLM model: {llm_model_path}")

    tlm_config = TLMConfig(
        llm_model_path=llm_model_path,
        freeze_ts_model=True,
        ts_pad_num=args.prefix_num,
        prefix_num=args.prefix_num,
        m_d_model=config.get('m_d_model', 896),
        m_n_heads=config.get('m_n_heads', 16),
        m_layers=config.get('m_layers', 2),
        m_dropout=config.get('m_dropout', 0.1),
        dropout=config.get('dropout', 0.1),
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8),
        e_layers=config.get('e_layers', 4),
        patch_len=config.get('patch_len', 60),
        stride=config.get('stride', 60),
        input_len=config.get('input_len', 600),
    )

    from models.TimeLanguageModel import TLM
    model = TLM.from_pretrained(args.model_checkpoint, config=tlm_config, ts_config=tlm_config)
    if accelerator.is_main_process:
        param_counts = count_model_parameters(model)
        print(f"\n🔢 Model parameter counts (in M):")
        print(f"   LLM:         {param_counts['llm']:.2f}M")
        print(f"   MFormer:    {param_counts['mformer']:.2f}M")
        print(f"   TS_Encoder:  {param_counts['ts_encoder']:.2f}M")
        print(f"   Total:       {param_counts['total']:.2f}M")
    # Prepare model with accelerator
    model = accelerator.prepare(model)
    model.eval()
    if accelerator.is_main_process:
        print(f"✅ Model loaded successfully!")
        print("\n📊 Preparing test dataset...")
    # Load tokenizer
    if os.path.exists(os.path.join(args.model_checkpoint, "tokenizer.json")):
        if accelerator.is_main_process:
            print("Loading tokenizer from checkpoint")
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    else:
        if accelerator.is_main_process:
            print("Loading tokenizer from original model")
        tokenizer = model.tokenizer
    if '<|image_pad|>' not in tokenizer.get_vocab():
        if accelerator.is_main_process:
            print("Adding <|image_pad|> token to tokenizer")
        tokenizer.add_tokens(['<|image_pad|>'])
        model.llm_model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = 'left'
    class SimpleConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    tlmconfig = SimpleConfig(ts_pad_num=args.prefix_num)
    test_dataset = TsQaDataset(
        args.ts_path_test,
        args.qa_path_test,
        tokenizer,
        tokenizer,  # Use tokenizer as processor
        tlmconfig
    )
    data_collator = DataCollator(tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers
    )
    test_loader = accelerator.prepare(test_loader)
    if accelerator.is_main_process:
        print(f"📁 Test set size: {len(test_dataset)} samples")
        print(f"🔢 Batch size: {args.batch_size}, Total batches: {len(test_loader)}")
        print("\n🔍 Starting test set inference...")
    results = []
    with torch.no_grad():
        if accelerator.is_main_process:
            batch_iterator = tqdm(test_loader, desc="Inference progress")
        else:
            batch_iterator = test_loader
        for batch_idx, batch in enumerate(batch_iterator):
            # if batch_idx ==16:
            #     break
            unwrapped_model = accelerator.unwrap_model(model)
            generated_ids = unwrapped_model.generate(
                input_ids=batch['input_ids'],
                query_ids=batch['query_ids'],
                ts_values=batch['ts_values'],
                stage=batch['stage'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                num_beams=1,                    # Greedy search, fastest
                temperature=1.0,                # Avoid extra computation
                top_p=None,                     # Disable nucleus sampling
                top_k=None,                     # Disable top-k sampling
                repetition_penalty=1.0,         # Disable repetition penalty
                length_penalty=1.0,             # Disable length penalty
                no_repeat_ngram_size=0,         # Disable n-gram repetition check
                output_scores=False,            # Do not output scores
                output_attentions=False,        # Do not output attention
                output_hidden_states=False,     # Do not output hidden states
                return_dict_in_generate=False,  # Simplify return format
            )
            batch_predictions = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            batch_labels = tokenizer.batch_decode(
                batch['labels'], 
                skip_special_tokens=True
            )
            for i in range(len(batch_predictions)):
                prediction = batch_predictions[i].split('assistant\n')[-1]
                results.append({
                    "index": batch['index'][i].item(),
                    "ts_id": batch['ts_id'][i],  # 🌟 新增的 ts_id 字段
                    "stage": batch['stage'][i].item(),
                    "input": tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True),
                    "prediction": prediction,
                    "label": batch_labels[i],
                    "is_correct": prediction.strip() == batch_labels[i].strip()
                })
        rank = accelerator.process_index
        temp_save_path = os.path.join(args.output_dir, f"temp_results_rank_{rank}.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(temp_save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            print("\n📊 Combining results from all GPUs (Safe Mode)...")
            all_combined_results = []

            for i in range(accelerator.num_processes):
                part_path = os.path.join(args.output_dir, f"temp_results_rank_{i}.json")
                if os.path.exists(part_path):
                    with open(part_path, 'r', encoding='utf-8') as f:
                        all_combined_results.extend(json.load(f))
                    os.remove(part_path)

            seen_indices = set()
            unique_results = []
            for r in all_combined_results:
                if r['index'] not in seen_indices:
                    unique_results.append(r)
                    seen_indices.add(r['index'])
            results = sorted(unique_results, key=lambda x: x['index'])

            print("📊 Calculating evaluation metrics...")

            metrics = compute_metrics_from_results(results, args)
            print_metrics(metrics)
            config_base = os.path.basename(args.config).split('.yaml')[0]
            save_results(results, args.output_dir, config_base)
            save_metrics(metrics, args.output_dir, config_base)

def compute_metrics_from_results(results: List[Dict],args) -> Dict[str, Any]:
    """Compute evaluation metrics for each stage from inference results."""
    stage1_data = [r for r in results if r['stage'] == 1]
    stage2_data = [r for r in results if r['stage'] == 2]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
    special_id = tokenizer.all_special_ids  
    common_punctuations = [".", ",", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}", "-", "_", "\"", "'"]
    punctuation_ids = tokenizer.convert_tokens_to_ids(common_punctuations)
    special_id.extend(punctuation_ids)
    metrics = {}
    if stage1_data:
        stage1_metrics = closed_question_metrics(
            [r['prediction'] for r in stage1_data],
            [r['label'] for r in stage1_data],
            special_id
        )
        metrics.update({f"stage_1_closed_{k}": v for k, v in stage1_metrics.items()})
    if stage2_data:
        stage2_metrics = open_question_metrics(
            [r['prediction'] for r in stage2_data],
            [r['label'] for r in stage2_data],
            special_id
        )
        metrics.update({f"stage_2_open_{k}": v for k, v in stage2_metrics.items()})
    return metrics

def print_metrics(metrics: Dict[str, Any]):
    print("\n📈 Evaluation results:")
    for stage, label in [(1, "Stage 1 (Closed)"), (2, "Stage 2 (Open)")]:
        stage_metrics = {k.replace(f"stage_{stage}_closed_", "").replace(f"stage_{stage}_open_", ""): v 
                        for k, v in metrics.items() if k.startswith(f"stage_{stage}_")}
        if stage_metrics:
            print(f"\n🔹 {label} metrics:")
            for metric, value in stage_metrics.items():
                print(f"   {metric}: {value:.4f}")

def save_metrics(metrics: Dict[str, Any], output_dir: str, config_base: str):
    metrics_file = os.path.join(output_dir, f"{config_base}_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Evaluation metrics saved to: {metrics_file}") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TLM Inference')
    parser.add_argument('--config', type=str, default='yaml/infer.yaml', help='YAML config file')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--model_checkpoint', type=str, default='checkpoints/former-0.5B', help='Model checkpoint path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum new tokens to generate')
    args = parser.parse_args()
    main_inference(args)