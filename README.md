# ITFormer

此资源库提供了 ITFormer（Instruct Time Transformer）的官方开源实现，ITFormer 是一种用于时间-文本多模态问答（QA）的新型框架。

## Overview

ITFormer（Instruct Time Transformer）是一款用于时态-文本多模态问答的顶尖模型。此资源库提供了官方的开源实现版本，其中包括推理和训练脚本。

我们的研究引入了一个大规模的多任务数据集([EngineMT-QA][EngineMT-QA])，并展示了 ITFormer 在将时间序列数据与自然语言理解相结合方面表现出的卓越性能。值得一提的是，我们的 0.5 亿参数模型体积小巧、运行高效，同时仍能取得出色的效果。

[EngineMT-QA]:https://huggingface.co/datasets/pandalin98/EngineMT-QA

## Features

- 📊 **预训练模型**：在 Hugging Face 上可直接使用的 ITFormer 模型([0.5B][ITFormer0.5B], [3B][ITFormer3B], [7B][ITFormer7B])。
- 🚀 **轻量且高效**：0.5 亿参数的模型具备强大的时态问答能力，并且易于部署。([Qwen2.5-Instruct][Qwen2.5-Instruct])
- 🎯 **一键式脚本**：提供用于预训练、SFT（序列到序列训练）和并行推理的自动化脚本。
- 📈 **高性能**：在时态文本问答基准测试中取得了最先进的结果。
- 🌐 **分布式支持**：与 `accelerate` 完全兼容，适用于多 GPU 训练和推理。

[ITFormer0.5B]:https://huggingface.co/pandalin98/ITFormer-0.5B
[ITFormer3B]:https://huggingface.co/pandalin98/ITFormer-3B
[ITFormer7B]:https://huggingface.co/pandalin98/ITFormer-7B
[Qwen2.5-Instruct]:https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

## Quick Start

### 1. Organize Directory Structure

下载模型和数据集后，请按照以下方式整理您的文件：

<pre>
ITFormer-ICML25/
├── dataset/
│   ├── dataset.py                   # EngineMT-QA的辅助文件
│   └── datasets/                    # 将 EngineMT-QA 数据集文件放置在此处
│       ├── time_series_data.h5
│       ├── train_qa.jsonl
│       └── test_qa.jsonl
│
├── assets/                          # 静态资源仓库
├── LLM/                             # 基础 Qwen2.5-Instruct 模型
│   └── Qwen2.5-0.5B-Instruct/
├── checkpoints/
│   └── ITFormer-0.5B/               # ITFormer model 的检查点
│
├── yaml/                            # 参数配置
│   ├── accelerate_config.yaml
│   └── infer.yaml                   
│
├── .venv_linux/                     # Linux 虚拟环境（python3.11）
├── requirements.txt                 # 该项目所依赖的所有库及其版本
│
├── models/                          # ITFormer的定义
│   ├── ITFormer.py
│   ├── TimeLanguageModel.py
│   ├── TimeSeriesEncoder.py
│   └── layers/
│       └── attention.py
├── utils/                          # 辅助函数
│   ├── dist_util.py                  # 分布式状态检测文件
│   ├── gradio_vlm.py                 # 用户界面组件
│   ├── log_util.py                   # 日志防刷屏文件
│   ├── metrics.py                    # 评测指标
│   └── position_coding.py            # 位置编码
├── EXP/                           # 实验逻辑
│   ├── exp_instruct.py               # 指令微调（SFT）：用于 Stage 1-4 的故障诊断任务微调
│   └── exp_pretraining.py            # 预训练：用于训练 TimeSeries Encoder（时间序列编码器）学习基础的物理特征
│           
├── scripts/                       # 一键式自动化脚本
│   ├── run_pretrain.sh
│   ├── run_sft.sh
│   └── run_inference.sh
├── train_pretrain.py              # 预训练文件
├── train_sft.py                   # sft训练文件
├── inference.py                   # inference推理文件
│
├── save/                          # 保存模型
│   ├── pretrain/                    # 训练好的TimeSeries Encoder
│   ├── pretrain_ts_small/           # TimeSeries Encoder的临时存档
│   └── sft_qwen2.5_0.5B/            # 训练好的ITFormer+LLM
├── swanlog/                       # 由 SwanLab 库生成的日志文件夹
│   └── run-xxx/
├── inference_results/             # Inference的结果
│   ├── inference_results.json       # 对于测试集的问题，模型的答案
│   └── infer_metrics.json           # 在 inference.py 产生的评价指标
├── metrics_eval.txt               # 在 train_sft.py 产生的评价指标
│
└── .gitignore                     # github的配置文件
</pre>

### 2. Environment Setup and Dependencies
使用 WSL2 (Ubuntu 22.04/24.04) 在 Windows 内部挂载 Linux 内核，以便配置分布式训练和推理。

在管理员权限的 PowerShell 中运行在管理员权限的 PowerShell 中运行
```
wsl --update
# 查看你当前系统中可用的在线分发版列表
wsl --list --online
# 如果列表里只有 Ubuntu，就用这个
wsl --install -d Ubuntu
```
---
按下键盘上的 Win 键，在搜索框里输入 “Ubuntu”。
```
# 查看WSL2与联Windows的显卡驱动的关联情况
nvidia-smi
# 准备venv 组件
sudo apt update
sudo apt install python3.11-venv -y
# 在linux 的用户主目录，创建一个存放项目的文件夹。把项目从 Windows 桌面拷贝过来 (注意路径)；(在 Ubuntu 中，Windows 的 C 盘挂载在 /mnt/c/)
cd ~
mkdir -p projects
cp -r /mnt/c/Users/cw/Desktop/ITFormer-ICML25-main ./itformer
cd itformer
# 创建虚拟环境
python3.11 -m venv .venv_linux
source .venv_linux/bin/activate
pip install -r requirements.txt
# 配置 Accelerate
accelerate config
```
Accelerate的配置选择

<img src="assets/img_0.png" width="800">

---

Windows中PyCharm中连接wsl中的虚拟环境

<img src="assets/img_1.png" width="800">

## Validation and Testing

修改后始终运行这些验证步骤：

```bash
# 1. 测试核心导入（快速验证）
python -c "from models.TimeLanguageModel import TLM, TLMConfig; from dataset.dataset import TsQaDataset; from utils.metrics import open_question_metrics; print('✅ All modules import successfully')"

# 2. 测试推理脚本帮助（验证参数解析）
python inference.py --help

# 3. 所有Python文件的语法验证
find . -name "*.py" -exec python -m py_compile {} \;

# 4. 测试基本模型初始化（需要下载模型）
python -c "
import yaml
with open('yaml/infer.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('✅ Configuration file loads correctly')
"
```

## Run Inference

我们支持使用“accelerate”库进行“并行推理”。这会自动汇总来自多个 GPU 的结果。

```bash
# 使用自动化脚本（推荐）
bash scripts/run_inference.sh

# 或者通过accelerate手动启动
accelerate launch --config_file yaml/accelerate_config.yaml inference.py --config yaml/infer.yaml

# 方法 1：直接使用 Python 进行执行（单个 GPU/单个 CPU）
python inference.py --config yaml/infer.yaml
# 方法 2：使用加速功能的多 GPU 设备（推荐用于生产环境）
accelerate launch --config_file yaml/accelerate_config.yaml inference.py --config yaml/infer.yaml

# 方法 3：使用所提供的脚本(.sh是旧版本，.bash是新版本)
bash scripts/infra.bash
bash scripts/run_inference.sh
```

该推理脚本将：
- 加载 ITFormer 及其对应的 Qwen2.5-Instruct 版本。
- 将数据分配到所有可用的 GPU 上。
- 对结果进行汇总并保存到 `inference_results/` 目录以及 `output_result_all.json` 文件中。


监控推理进度：
```bash
# 查看输出目录的结果
watch ls -la inference_results/

# 监控GPU使用情况（如果可用）
nvidia-smi -l 5
```
## Training

我们使用“accelerate”库提供了一套简洁高效的训练流程。请确保您的“accelerate_config.yaml”文件已针对您的硬件进行了正确配置。

### A. Pre-training (Time-Series Encoder)

阶段 A 的重点是使用掩码建模对`TimeSeriesEncoder`进行预训练。

```bash
# 一键预训练
bash scripts/run_pretrain.sh
```

### B. Supervised Fine-Tuning (SFT)

阶段 B 执行端到端的SFT，通过ITFormer将TimeSeriesEncoder与LLM桥接。
```bash
# # 单步 SFT（需预先加载 ts_encoder 的权重）
bash scripts/run_sft.sh
```

**SFT 中的关键参数：**
- `--it_d_model`、`--it_n_heads`、`--it_layers`：这是 ITFormer 模块的配置参数。
- `--load_ts_encoder`：指向在阶段 A 中生成的权重文件的路径。
- `--llm_model_path`：指向基础 Qwen2.5-Instruct 模型的路径。

---

## Configuration

主配置文件： yaml/infer.yaml

你可能需要修改的关键参数：
- ts_path_test : 时间序列数据文件路径
- qa_path_test ：QA 对文件的路径
- batch_size : 推理批次大小（默认：12）
- 与 GPU 相关的设置由 accelerate_config.yaml 处理

## Key Components 

核心模块位置：

- inference.py : 主推理脚本和入口
- models/TimeLanguageModel.py : 核心 ITFormer 模型实现
- models/TimeSeriesEncoder.py : 时间序列编码组件
- dataset/dataset.py : 数据加载和预处理
- utils/metrics.py : 评估问答性能的指标
- yaml/infer.yaml : 主配置文件
- yaml/accelerate_config.yaml : 多 GPU 加速设置

模型架构：ITFormer 结合：
- Time series encoder时间序列编码器（处理时序数据）
- Large language model大型语言模型 (Qwen2.5 variants)
- Cross-modal attention mechanisms跨模态注意力机制
- Question-answering head for generation生成式问答头

## Performance Expectations

ITFormer 模型性能：
- 0.5B 模型：在时间问答方面超越 ChatGPT-4o，推理速度最快
- 3B 模型：平衡性能与速度
- 7B 模型：最佳准确率，最慢推理

资源需求：
- 最低 RAM：8GB（适用于 0.5B 模型）
- 推荐 RAM：16GB+（适用于 3B+）
- GPU 显存：4GB+（0.5B），12GB+（7B）
- 存储空间：所有模型和数据集需 50GB+

## Development Workflow

1. 始终首先验证导入 - 运行导入验证命令
2. 在尝试较大模型前先用最小模型（0.5B）进行测试
3. 当 GPU 资源有限时使用 CPU 模式进行测试
4. 监控推理运行期间的内存使用情况
5. 检查生成的结果和指标目录 inference_results/
6. 验证 yaml/infer.yaml 中的配置路径是否与您的目录结构匹配

## Model Architecture

ITFormer利用指令感知时间序列转换器**将时间特征与文本查询对齐，然后将它们输入到大型语言模型中。该框架被设计为参数高效，在SFT期间冻结LLM和TS编码器，同时只训练ITFormer和投影层。
## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{wang2025itformer,
  title={ITFormer: Bridging Time Series and Natural Language for Multi-Modal QA with Large-Scale Multitask Dataset},
  author={Yilin Wang and Peixuan Lei and Jie Song and Yuzhe Hao and Tao Chen and Yuxuan Zhang and Lei Jia and Yuanxiang Li and Zhongyu Wei},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## License

MIT License — see the LICENSE file for details.
