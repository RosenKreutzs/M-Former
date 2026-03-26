# 隐藏不必要的警告，确保在分布式训练时控制台不会被刷屏
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*TRANSFORMERS_CACHE.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModel
from dataset.dataset import TsQaDataset,DataCollator
import argparse
from models.TimeLanguageModel import TLMConfig, TLM
import os
import swanlab as wandb
from EXP.exp_instruct import Exp_Instruct

# 初始化加速器，它会自动处理设备分配（GPU/CPU）。
from accelerate import Accelerator
accelerator = Accelerator(device_placement=True)# # 限制只使用 GPU 0,debug模式


import os
import random
import numpy as np
import sys
import logging
from transformers.utils import logging as transformers_logging

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"#防止分词器在多线程中产生死锁。
os.environ["WANDB_MODE"] = "offline" #将实验记录工具（SwanLab/WandB）设为离线，方便在无网环境或私有集群运行。

# 设置日志级别，主进程显示进度，从进程静默
if os.environ.get("LOCAL_RANK", "0") == "0":
    transformers_logging.set_verbosity_info()
else:
    transformers_logging.set_verbosity_error()
    logging.disable(logging.CRITICAL)

# # 启用异常检测

if __name__ == '__main__':
    #读取args
    parser = argparse.ArgumentParser(description='Mutimodal SFT')
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')

    # TSE设置
    parser.add_argument('--model', type=str, required=False, default='TimeSeriesEncoder',help='model name')
    parser.add_argument('--d_model', type=int, default=512,help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4,help='num of encoder layers')
    parser.add_argument("--patch_len", type=int, default=60)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--input_len", type=int, default=600)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--load_ts_encoder', type=str, default='save/pretrain/model.safetensors', help='load ts_encoder')

    # ITFormer设置
    parser.add_argument('--it_d_model', type=int, default=896, help='dimension of IT model')
    parser.add_argument('--it_n_heads', type=int, default=16, help='num of IT heads')
    parser.add_argument('--it_layers', type=int, default=2, help='num of IT layers')
    parser.add_argument('--it_dropout', type=float, default=0.1, help='dropout for IT model')
    parser.add_argument('--prefix_num', type=int, default=25, help='number of prefixes')

    # LLM设置
    parser.add_argument('--llm_model_path', type=str, default='LLM/Qwen2.5-0.5B-Instruct', help='LLM model path')

    # 预训练的设置
    parser.add_argument('--pretrain', type=bool, default=False, help='pretrain mode')# 预训练模式开关。
    parser.add_argument('--min_mask_ratio', type=float, default=0.7, help='minimum mask ratio')# 最小遮盖比例。
    parser.add_argument('--max_mask_ratio', type=float, default=0.8, help='maximum mask ratio')# 最大遮盖比例。

    # 训练设置
    parser.add_argument('--do_train', type=bool, default=True, help='whether to do training')# 训练模式开关。
    parser.add_argument('--per_device_train_batch_size', type=int, default=12, help='batch size per device during training')# 训练阶段：每一块显卡一次“吞下”的数据量。
    parser.add_argument('--per_device_eval_batch_size', type=int, default=12, help='batch size for evaluation')# 验证阶段：每一块显卡一次“吞下”的数据量。
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate') # 学习率
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='number of updates steps to accumulate before performing a backward/update pass')# 梯度累加步数。
    parser.add_argument('--num_train_epochs', type=int, default=2, help='number of training epochs')# 训练轮数。
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')# 权重衰减。
    parser.add_argument('--freeze_ts_model',type=bool,default=True,help='wheter freeze ts encoder')# 冻结TSE的参数
    # 效率设置
    parser.add_argument('--fp16', type=bool, default=True, help='whether to use 16-bit (mixed) precision')# 半精度训练。把原本 32 位的计算精简为 16 位。
    parser.add_argument('--dataloader_pin_memory', type=bool, default=True, help='pin memory in data loader')# 把数据锁在内存中，让 CPU 往 GPU 传输数据时走“绿色通道”
    parser.add_argument('--dataloader_num_workers', type=int, default=4, help='number of subprocesses to use for data loading')# 雇佣 4 个多线程助手来提前读取数据。这样当显卡算完一组时，下一组数据已经准备好了，显卡不用“待岗”。

    # 日志设置
    parser.add_argument('--output_dir', type=str, default='save/sft_qwen2.5_0.5B_infra', help='output directory')# 用于保存成果和监控状态的文件夹
    parser.add_argument('--save_steps', type=int, default=1000, help='save checkpoint every X updates steps')# 每跑 1000 步就存一次档
    parser.add_argument('--save_total_limit', type=int, default=10, help='limit the total amount of checkpoints')# 存档上限。硬盘空间有限，只保留最近的 10 个模型包，旧的会自动删掉。
    parser.add_argument('--logging_steps', type=int, default=50, help='log every X updates steps')# 每 50 步汇报一次战况（当前的 Loss 是多少），这些数据会发送到你指定的 SwanLab 看板。
    parser.add_argument('--eval_steps', type=int, default=300000000000000000, help='eval every X updates steps')# 这里设了一个天文数字，实际上是为了关掉训练过程中的自动评估。在训练全部结束（通过脚本最后的 Trainer.evaluate()）
    parser.add_argument('--report_to', type=str, default="swandb", help='report results to')# 可视化工具。你可以直接在网页浏览器里看到 Loss 曲线是不是在平稳下降。
    parser.add_argument('--mode', type=str, default='train', help='inference or train')# 任务模式。
    parser.add_argument('--eval_stragy',type=str,default="no",help='The evaluation strategy to adopt during training')# 设置为 no 意味着在每一轮（Epoch）结束时都不进行自动测试。这与上面的 eval_steps 配合，确保训练过程全速前进，不被打断。
    parser.add_argument('--shuffle', type=bool, default=True, help='whether to shuffle the dataset')# 在训练开始前，打乱数据集的顺序。

    # 使用parser更新args
    args = parser.parse_args()

    # 设置固定的随机种子
    seed = 42

    # Python 随机模块
    random.seed(seed)

    # NumPy 随机模块
    np.random.seed(seed)

    # PyTorch 随机模块
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 对 CUDA 的种子进行控制
    torch.cuda.manual_seed_all(seed)  # 对所有 GPU 进行控制
    # 模型设置
    tlmconfig = TLMConfig(llm_model_path = args.llm_model_path,freeze_ts_model=args.freeze_ts_model,
                          ts_pad_num=args.prefix_num)
    # 数据集设置
    ts_past_train = 'dataset/datasets/train_data.h5'
    qa_past_train = 'dataset/datasets/train_qa.jsonl'
    ts_path_test = 'dataset/datasets/test_data.h5'
    qa_path_test = 'dataset/datasets/test_qa.jsonl'

    tokenizer = AutoTokenizer.from_pretrained(tlmconfig.llm_model_path)
    tokenizer.padding_side = 'left'
    processor = AutoProcessor.from_pretrained(tlmconfig.llm_model_path)
    train_dataset = TsQaDataset(ts_past_train, qa_past_train, tokenizer, processor, tlmconfig,sft=True, shuffle=args.shuffle)
    test_dataset = TsQaDataset(ts_path_test, qa_path_test,tokenizer, processor, tlmconfig)

    # 监控初始化（防止重复记录）
    if accelerator.is_main_process:
        import swanlab as wandb
        # wandb.init(project="TSLLM", name="pandalin")
        #设置offline
        wandb.init(mode="offline", project="XXX", name="XXX")

    Trainer = Exp_Instruct(args, train_dataset=train_dataset, eval_dataset=test_dataset,tlm_config=tlmconfig)       # 创建训练器
    Trainer.train(resume_from_checkpoint=False) # （从零开始）启动训练过程。
    Trainer.evaluate() # 训练结束后，立即在测试集上运行一遍，输出 BLEU、Accuracy 等指标 。

    
    

    
    