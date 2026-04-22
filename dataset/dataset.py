#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时间序列问答数据集。
处理时间序列数据和问答对的加载和预处理。
"""
import sys
from transformers import PretrainedConfig, AutoTokenizer
from transformers import AutoProcessor
import torch
import os
import json
from torch.utils.data import Dataset
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import h5py
import re
import random
from models.TimeLanguageModel import TLMConfig
from utils.log_util import adaptive_print
import pywt  # 🌟 引入专业的连续小波变换库
import scipy.signal as signal
"""
PretrainDataset 预训练加载器（只处理时间序列信号）:从.h5文件中提出seq_data，并转化为张量输出。
"""


class PretrainDataset(Dataset):

    def __init__(self, ts_path,target_f=64, target_t=64, return_cwt=False):
        super().__init__()
        self.ts_path = ts_path
        self.target_f = target_f
        self.target_t = target_t
        self.return_cwt = return_cwt
        self.load_data()

    def load_data(self):
        with h5py.File(self.ts_path, 'r') as f:
            data = f['seq_data'][:]
        self.datas = data

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        ts = torch.tensor(self.datas[index], dtype=torch.float)
        return {'ts_values':ts}


"""
find_assistant_tokens SFT训练加载器（QA处理）:从QA序列中识别特定的标识符，并返回位置坐标。（为了掩盖非assistant序列，只算assistant序列的损失）
"""


def find_assistant_tokens(tokenizer, target):
    """Find assistant token positions in the target sequence.

    Args:
        tokenizer: Tokenizer instance
        target: Target token sequence

    Returns:
        List of tuples containing start and end positions of assistant tokens
    """
    result = []
    start_index = 0
    end_index = 0
    while start_index <= len(target) - 1:
        if target[start_index] != tokenizer('assistant')['input_ids'][0]:
            start_index += 1
            end_index += 1
        else:
            end_index += 1
            if target[end_index] == tokenizer('<|im_end|>')['input_ids'][0]:
                result.append((start_index + 1, end_index + 1))
                start_index = end_index + 1
    return result


class TsQaDataset(Dataset):
    """TsQaDataset 数据集类：具有令牌ID范围验证的时间序列问答数据集。"""

    def __init__(self, ts_path, data_path, tokenizer, processor, config, target_f=64, target_t=64, return_cwt=False,pretrain=False, sft=False, shuffle=False):
        """Initialize the dataset.

        Args:
            ts_path: 信号数据的路径
            data_path: 问答对文件的路径
            tokenizer: 分词器对象
            processor: 多模态处理器对象
            config: config对象
            pretrain: pretrain模式开关
            sft: sft
            shuffle: 决定是否在索引构建完成后立即打乱数据
        """
        super().__init__()  # 调用父类 torch.utils.data.Dataset 的初始化方法。
        self.ts_path = ts_path
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.pretrain = pretrain
        self.sft = sft
        self.shuffle = shuffle
        self.h5_file = None
        self.target_f = target_f
        self.target_t = target_t
        self.return_cwt = return_cwt

        # 获取分词器的词表大小
        self.vocab_size = len(self.tokenizer)
        adaptive_print(f"📊 Vocab size: {self.vocab_size}")

        # 强制设置为左填充。wen2.5 是 Decoder-only（仅解码器）架构。在推理生成答案时，模型是从左往右读、向右方吐字的。如果使用右填充，填充的补位 Token 会干扰位置编码，导致模型生成的文字变成乱码。
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._validate_special_tokens()  # 立即执行专项检查。它会确认分词器的 pad、eos 等特殊 ID 是否真的在 [0, vocab_size) 范围内。这主要是为了修复某些版本分词器在合并自定义 Token 后可能出现的 ID 溢出 Bug。
        self._build_index()  # 描你的 train_qa.jsonl 文件，根据你之前关心的 stage（1-4 阶段）过滤出有效数据，并记录每条问答在 H5 文件中对应的信号 ID。

    def __len__(self):
        """__len__: 标准的 PyTorch Dataset 方法。返回数据集中有效样本的总数（即经过 _build_index 筛选后的条目数）。"""
        return len(self.datas)

    def __del__(self):
        """__del__: 析构函数。在对象销毁时确保关闭 HDF5 文件句柄，防止内存泄漏或文件锁定。"""
        if self.h5_file:
            self.h5_file.close()

    def _validate_special_tokens(self):
        """_validate_special_tokens: 启动检查。核实 pad、eos、bos 等特殊 Token ID 是否在词表范围内。如果发现 pad_token_id 非法，会将其自动修复为 eos_token_id 以防止 Embedding 层报错。"""
        special_tokens = {
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'bos_token_id': getattr(self.tokenizer, 'bos_token_id', None),
            'unk_token_id': getattr(self.tokenizer, 'unk_token_id', None),
        }

        adaptive_print("🔍 Validating special tokens:")
        for name, token_id in special_tokens.items():
            if token_id is not None:
                if token_id >= self.vocab_size or token_id < 0:
                    adaptive_print(f"❌ {name} = {token_id} out of range [0, {self.vocab_size})")
                    # Fix invalid special tokens
                    if name == 'pad_token_id':
                        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                        adaptive_print(f"🔧 Fixed: pad_token_id -> {self.tokenizer.pad_token_id}")
                else:
                    adaptive_print(f"✅ {name} = {token_id}")

    def _validate_token_ids(self, token_ids, context=""):
        """_validate_token_ids: 序列过滤器。遍历一组 Token ID，若发现任何 ID 超出词表范围（小于 0 或大于等于 vocab_size），则将其替换为 unk_token_id（或 eos_token_id），确保输入模型的 ID 绝对安全。"""
        if not isinstance(token_ids, list):
            return token_ids

        valid_ids = []
        for i, token_id in enumerate(token_ids):
            if token_id < 0 or token_id >= self.vocab_size:
                adaptive_print(f"⚠️ {context} position {i}: invalid token_id {token_id}, replacing with unk_token")
                # Replace with unk_token, if not available use eos_token
                replacement = getattr(self.tokenizer, 'unk_token_id', self.tokenizer.eos_token_id)
                valid_ids.append(replacement)
            else:
                valid_ids.append(token_id)
        return valid_ids

    def _build_index(self):
        """_build_index: 数据扫描器。它会完整读取 .jsonl 文件，解析每一行对话。只有 stage 为 '1'（封闭式任务）或 '2'（开放式任务）的样本会被提取并存入 self.datas 列表。如果开启了 shuffle，它还会打乱样本顺序。"""
        self.datas = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                item = json.loads(line)
                for i in range(0, len(item['conversations']), 2):
                    if item['conversations'][i]['stage'] in ['1', '2']:
                        self.datas.append({
                            'id': item['id'],
                            'stage': int(item['conversations'][i]['stage']),
                            'form': item['conversations'][i]['attribute'],
                            'question': item['conversations'][i]['value'],
                            'answer': item['conversations'][i + 1]['value'],
                            'line_num': line_num
                        })

        h5f = self._get_h5_file()  # 将 H5 里的字节型 ID 转换为字符串列表
        all_ids = [x.decode('utf8') for x in h5f['data_ID'][:]]
        self.id_to_idx = {str(sid): i for i, sid in enumerate(all_ids)}  # 建立 原始ID -> H5物理行索引 的字典
        if self.shuffle:
            adaptive_print(f"🎲 Shuffling dataset: {self.data_path}")
            random.shuffle(self.datas)

    def _get_h5_file(self):
        """_get_h5_file: 懒加载管理器。它不会在初始化时立刻打开巨大的 H5 文件，而是在第一次需要读取信号数据时才建立连接。"""
        if self.h5_file is None and os.path.exists(self.ts_path):
            self.h5_file = h5py.File(self.ts_path, 'r')
        return self.h5_file

    def add_adaptive_prompt(self, sample):
        """add_adaptive_prompt: 指令增强器。根据任务阶段（1-4）为问题追加特定的引导语。例如，为 Stage 4 追加“请提出具体的维修建议”，帮助模型理解其作为“决策者”的角色。"""
        sample = sample.copy()

        if sample['stage'] == 1:
            sample[
                'question'] += " Please analyze the change in this signal and explain its physical implication, such as component load, airflow, or temperature stability."
        elif sample['stage'] == 2:
            sample[
                'question'] += " Carefully analyze the signal pattern (e.g., stability, oscillation, drops) to determine the correct fault status or root cause. Select the most likely option based on observed signal behavior."
        elif sample['stage'] == 3:
            sample[
                'question'] += " Review the trends across 10 cycles and evaluate the degradation pattern. Select the option that best reflects the long-term health status or risk level indicated by the signal."
        elif sample['stage'] == 4:
            sample[
                'question'] += " Based on the 10-cycle degradation pattern, propose concrete maintenance actions (e.g., replace, inspect) to ensure safe and efficient operation."
        return sample

    def _create_chat_input(self, question):
        """_create_chat_input: 模版缝合器。它使用模型的聊天模版（Chat Template）封装问题，并将文本中的占位符 <ts> 替换为指定数量的 <|image_pad|> Token。这些位置随后会被 MFormer 生成的信号特征填补。"""
        messages = [
            {"role": "system", "content": 'You are a helpful assistant.'},
            {"role": "user", "content": question}
        ]

        try:
            # Use a safer tokenization method
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Replace time series placeholder
            chat_text = chat_text.replace('<ts>', '<|image_pad|>' * self.config.ts_pad_num)
            return chat_text
        except Exception as e:
            adaptive_print(f"❌ Chat template error: {e}")
            # Fallback to a simple format
            return f"You are a helpful assistant.\nuser\n{question}\nassistant\n"

    def _safe_tokenize(self, text, add_special_tokens=True):
        """_safe_tokenize: 安全分词器。它在调用标准分词逻辑后，会自动运行"""
        try:
            # Add more tokenization parameters
            result = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                padding=False,
                truncation=False,
                return_tensors=None
            )
            token_ids = result['input_ids']

            # Validate token_ids
            token_ids = self._validate_token_ids(token_ids, f"tokenize: {text[:50]}...")
            return token_ids

        except Exception as e:
            adaptive_print(f"❌ Tokenization error for text: {text[:100]}...")
            adaptive_print(f"Error: {e}")
            # Return a safe default value
            return [self.tokenizer.eos_token_id]

    def __getitem__(self, idx):
        """__getitem__: 核心调度函数。"""
        try:
            # 索引中取出该条问答的元数据（包含信号 ID、问题、答案等）。
            sample = self.datas[idx]
            # sample = self.add_adaptive_prompt(sample)

            # 加载时序数据
            h5f = self._get_h5_file()

            def get_signal_by_id(raw_id):
                sid = str(raw_id)
                if sid not in self.id_to_idx:
                    raise KeyError(f"ID {sid} not found in H5 data_ID")
                actual_idx = self.id_to_idx[sid]
                signal = h5f['seq_data'][actual_idx]
                # 强制确保形状为 (500, 2)，防止出现 [100] 或其他降维形状
                return signal.reshape(500, 2)

            # 新设计：所有任务均基于单周期信号，id 字段为列表格式 ["1"]，取第一个元素
            if isinstance(sample['id'], list):
                ts = get_signal_by_id(sample['id'][0])
            else:
                # 向后兼容：id 为字符串或整数时直接使用
                ts = get_signal_by_id(sample['id'])

            # =========================== 模式 1: Pretraining ===========================
            if getattr(self, 'pretrain', False):
                ts = torch.tensor(ts, dtype=torch.float)
                return {'ts_values': ts}

            # =========================== 模式 2: SFT Training ===========================
            elif getattr(self, 'sft', False):
                # 对纯问题文本进行分词。这主要用于后续评估或对比
                original_question = sample['question']
                query_ids = self._safe_tokenize(original_question, add_special_tokens=False)

                # 在问题中插入 prefix_num=25 个 <|image_pad|> 占位符。
                q_text = self._create_chat_input(sample['question'])  # This includes <|image_pad|>
                q_input_ids = self._safe_tokenize(q_text, add_special_tokens=False)

                # 对答案分词，并强制在末尾加上 eos_token（终止符），告诉模型什么时候该停止说话。
                a_text = sample['answer']
                if not a_text.endswith(self.tokenizer.eos_token):
                    a_text += self.tokenizer.eos_token
                a_input_ids = self._safe_tokenize(a_text, add_special_tokens=False)

                # Construct training data
                input_ids = q_input_ids + a_input_ids
                labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids

                # 验证token_id
                query_ids = self._validate_token_ids(query_ids, f"query_ids_sample_{idx}")
                input_ids = self._validate_token_ids(input_ids, f"final_input_sample_{idx}")  # input_ids = 问题 + 答案。
                labels = self._validate_token_ids(labels,
                                                  f"final_labels_sample_{idx}")  # labels = [Pad ID] * 问题长度 + 答案。；

                # 确认长度（错位对齐）：这是自回归训练的标准操作。用第 t 个词去预测第 t+1 个词
                final_input_ids = input_ids[:-1] if len(input_ids) > 1 else input_ids
                final_labels = labels[1:] if len(labels) > 1 else labels

                ts = torch.tensor(ts, dtype=torch.float)

                return {
                    'form': sample['form'],
                    'stage': sample['stage'],
                    'query_ids': query_ids,  # Only contains the original question text
                    'input_ids': final_input_ids,
                    'labels': final_labels,
                    'ts_values': ts,
                    'index': sample['line_num'],
                    'ts_id': str(sample['id'])  # 🌟 新增的键值对
                }

            # =========================== 模式 3: Inference/Evaluation ===========================
            else:  # 在推理时，我们不能把答案拼在问题后面。
                # 问题文本分词
                original_question = sample['question']
                query_ids = self._safe_tokenize(original_question, add_special_tokens=False)

                # 在问题中插入 prefix_num=25 个 <|image_pad|> 占位符。
                q_text = self._create_chat_input(sample['question'])
                q_input_ids = self._safe_tokenize(q_text, add_special_tokens=False)

                # 答案文本分词
                a_text = sample['answer']
                if not a_text.endswith(self.tokenizer.eos_token):
                    a_text += self.tokenizer.eos_token
                a_input_ids = self._safe_tokenize(a_text, add_special_tokens=False)

                # 验证token_ids
                query_ids = self._validate_token_ids(query_ids, f"infer_query_sample_{idx}")
                q_input_ids = self._validate_token_ids(q_input_ids, f"infer_q_sample_{idx}")
                a_input_ids = self._validate_token_ids(a_input_ids, f"infer_a_sample_{idx}")

                ts = torch.tensor(ts, dtype=torch.float)

                return {
                    'form': sample['form'],
                    'stage': sample['stage'],
                    'query_ids': query_ids,  # Only contains the original question text
                    'input_ids': q_input_ids,
                    'labels': a_input_ids,
                    'ts_values': ts,
                    'index': sample['line_num'],
                    'ts_id': str(sample['id'])  # 🌟 新增的键值对
                }

        except Exception as e:
            # 异常处理机制
            adaptive_print(f"❌ Error processing sample {idx}: {e}")
            # 如果在读取 H5 文件或分词过程中某一行数据坏了，会返回一个“全零信号 + 终止符文本”的占位数据
            return self._get_safe_default_sample()
    def _get_safe_default_sample(self):
        """_get_safe_default_sample: 容错机制。如果 __getitem__ 在处理某个样本时崩溃，它会返回一个由零张量和特殊 Token 组成的“安全样板”，确保训练循环不会中断。"""
        return {
            'form': 'default',
            'stage': 1,
            'query_ids': [self.tokenizer.eos_token_id],
            'input_ids': [self.tokenizer.eos_token_id],
            'labels': [self.tokenizer.eos_token_id],
            'ts_values': torch.zeros((500, 2), dtype=torch.float),
            'index': 0,
            'ts_id': 'default'  # 🌟 补齐容错数据结构
        }

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 分词器的左填充校准
        if self.tokenizer.padding_side != 'left':
            adaptive_print("⚠️  Warning: Setting tokenizer.padding_side to 'left' for decoder-only model")
            self.tokenizer.padding_side = 'left'

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """__call__:多模式兼容逻辑。当 DataLoader 准备好一组样本（features）时，会自动调用这个方法。"""
        # 兼容性处理：检查 key 是否存在，如果不存在（如预训练模式）则跳过相关逻辑
        has_text_data = all('input_ids' in f and 'labels' in f and 'query_ids' in f for f in features)
        if not has_text_data:
            # 预训练模式或数据缺失：仅处理 ts_values
            ts_values = [f['ts_values'] for f in features]
            ts_values = torch.stack(ts_values, dim=0)
            batch = {'ts_values': ts_values}

            # 尝试包含其他可能存在的张量字段
            for key in ['stage', 'index']:
                if all(key in f for f in features):
                    batch[key] = torch.tensor([f[key] for f in features])

            # 🌟 新增：处理字符串类型的 ts_id，直接收集为 List[str]
            if all('ts_id' in f for f in features):
                batch['ts_id'] = [f['ts_id'] for f in features]

            return batch

        # 动态长度计算：由于同一个 Batch 中每个句子的长度不同，为了把它们放进矩阵，必须以本 Batch 中最长的那个句子为标准进行补齐。
        max_len_inputs = max(len(feature['input_ids']) for feature in features)
        max_len_labels = max(len(feature['labels']) for feature in features)
        max_len_querys = max(len(feature['query_ids']) for feature in features)

        # 样本循环处理与补齐
        input_ids = []
        attention_mask = []
        labels = []
        ts_values = []
        stages = []
        index = []
        query_ids = []
        ts_ids = []  # 🌟 新增的 ts_id 收集器

        for feature in features:
            input_len = len(feature['input_ids'])
            label_len = len(feature['labels'])
            query_ids_len = len(feature['query_ids'])

            # 左填充处理
            padded_input = [self.tokenizer.pad_token_id] * (max_len_inputs - input_len) + feature['input_ids']
            input_ids.append(padded_input)

            # Pad 的位置设为 0，真实 Token 的位置设为 1
            attention_mask.append([0] * (max_len_inputs - input_len) + [1] * input_len)

            # 使用 pad_token_id 保证在 loss 忽略 pad 位置
            padded_labels = [self.tokenizer.pad_token_id] * (max_len_labels - label_len) + feature['labels']
            labels.append(padded_labels)

            # 问题的 token_id 也左填充
            padded_query_ids = [self.tokenizer.pad_token_id] * (max_len_querys - query_ids_len) + feature['query_ids']
            query_ids.append(padded_query_ids)

            # 收集标量和张量
            ts_values.append(feature['ts_values'])
            stages.append(feature['stage'])
            index.append(feature['index'])

            # 🌟 收集 ts_id，加入容错：若没有提取到则返回 'default'
            ts_ids.append(feature.get('ts_id', 'default'))

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'ts_values': torch.stack(ts_values, dim=0),  # 将所有信号样本堆叠
            'stage': torch.tensor(stages, dtype=torch.int8),
            'index': torch.tensor(index, dtype=torch.int32),
            'query_ids': torch.tensor(query_ids, dtype=torch.long),
            'ts_id': ts_ids  # 🌟 直接作为 List[str] 放入批次字典
        }

