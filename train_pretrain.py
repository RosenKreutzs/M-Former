# 隐藏不必要的警告，确保在分布式训练时控制台不会被刷屏
import warnings
warnings.simplefilter("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*TRANSFORMERS_CACHE.*")

# 利用环境变量 LOCAL_RANK 判断当前进程。在多显卡训练时，只有主进程（rank 0）会输出详细日志，其他进程保持安静，避免重复打印。
import os
import logging
from transformers.utils import logging as transformers_logging
    # 只在主进程显示进度条和日志
if os.environ.get("LOCAL_RANK", "0") == "0":
    transformers_logging.set_verbosity_info()
else:
    transformers_logging.set_verbosity_error()
    logging.disable(logging.CRITICAL)


from transformers import  AutoTokenizer
from transformers import AutoProcessor
import torch.nn as nn
from transformers import Trainer
from dataset.dataset import TsQaDataset,PretrainDataset
import argparse
from models.TimeLanguageModel import TLMConfig
import swanlab as wandb
from EXP.exp_pretraining import Exp_Pretrain
from accelerate import Accelerator

if __name__ == '__main__':
    accelerator = Accelerator()
    
    # 读取args创建parser，用于定义模型设置，预训练设置，训练与效率设置。（fix_seed用于锚定的随机种子）
    parser = argparse.ArgumentParser(description='TsEncoder Pretrain')
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')

    # 定义模型设置
    parser.add_argument('--model', type=str, required=False, default='TimeSeriesEncoder',help='model name')
    parser.add_argument('--d_model', type=int, default=512,help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4,help='num of encoder layers')
    parser.add_argument("--patch_len", type=int, default=60)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--input_len", type=int, default=600)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    # 预训练设置
    parser.add_argument('--pretrain', type=bool, default=True, help='pretrain mode')# 预训练模式开关。
    parser.add_argument('--min_mask_ratio', type=float, default=0.7, help='minimum mask ratio')# 最小遮盖比例。
    parser.add_argument('--max_mask_ratio', type=float, default=0.8, help='maximum mask ratio')# 最大遮盖比例。

    # 训练设置
    parser.add_argument('--do_train', type=bool, default=True, help='whether to do training')# 训练模式开关。
    parser.add_argument('--per_device_train_batch_size', type=int, default=12, help='batch size per device during training')# 训练阶段：每一块显卡一次“吞下”的数据量。
    parser.add_argument('--per_device_eval_batch_size', type=int, default=12, help='batch size for evaluation')# 验证阶段：每一块显卡一次“吞下”的数据量。
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')# 学习率
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='number of updates steps to accumulate before performing a backward/update pass')# 梯度累加步数。
    parser.add_argument('--num_train_epochs', type=int, default=10, help='number of training epochs')# 训练轮数。
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')# 权重衰减。

    # 效率设置
    parser.add_argument('--fp16', type=bool, default=True, help='whether to use 16-bit (mixed) precision')# 半精度训练。把原本 32 位的计算精简为 16 位。
    parser.add_argument('--dataloader_pin_memory', type=bool, default=True, help='pin memory in data loader')# 把数据锁在内存中，让 CPU 往 GPU 传输数据时走“绿色通道”
    parser.add_argument('--dataloader_num_workers', type=int, default=8, help='number of subprocesses to use for data loading')# 雇佣 8 个多线程助手来提前读取数据。这样当显卡算完一组时，下一组数据已经准备好了，显卡不用“待岗”。

    # 日志设置
    parser.add_argument('--output_dir', type=str, default='save/pretrain_ts_small', help='output directory')# 用于保存成果和监控状态的文件夹
    parser.add_argument('--save_steps', type=int, default=100, help='save checkpoint every X updates steps')# 每跑 100 步就存一次档
    parser.add_argument('--save_total_limit', type=int, default=2, help='limit the total amount of checkpoints')#存档上限。硬盘空间有限，只保留最近的 2 个模型包，旧的会自动删掉。
    parser.add_argument('--logging_steps', type=int, default=10, help='log every X updates steps')# 每 10 步汇报一次战况（当前的 Loss 是多少），这些数据会发送到你指定的 SwanLab 看板。
    parser.add_argument('--report_to', type=str, default="swanlab", help='report results to')# 可视化工具。你可以直接在网页浏览器里看到 Loss 曲线是不是在平稳下降。

    # 使用parser更新args
    args = parser.parse_args()

    # 数据集设置
    ts_path = 'dataset/datasets/train_data.h5'
    dataset = PretrainDataset(ts_path)

    # 监控初始化（防止重复记录）
    if accelerator.is_main_process:
        wandb.init(mode="offline",project="TSLLM-TsEncoder", name="XXX")# 初始化实验追踪工具（将实验归类到这个项目目录下）

    Trainer = Exp_Pretrain(args, dataset)# 创建训练器

    Trainer.train(resume_from_checkpoint=False)# （从零开始）启动训练过程。
    Trainer.save_model('save/pretrain')#保存学习到的模型参数。
    Trainer.save_state()# 保存训练器的状态便于，接着训练。