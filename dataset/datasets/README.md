---
license: mit
task_categories:
- question-answering
pretty_name: PipelineManifold-QA
size_categories:
- 100K<n<1M
---


# PipelineManifold-QA Dataset

## Overview

**PipelineManifold-QA** 是一个多任务、多模态的数据集，用于**时间序列问答（Time-Series QA）**。它通过四个关键的认知任务，支持对多变量时间序列信号与自然语言进行对齐的研究：
**Pipeline Manifold-QA** 是一个面向工业管道多相流监控的多任务，多模态数据集。它旨在通过 **流形学习（Manifold Learning）** 与大语言模型的结合，实现对管道内部复杂流型（如气泡流、段塞流、环状流等）的自动识别与合理解释。该数据集将电压、电导等传感器信号序列与自然语言对齐，支持对工业流体动力学进行跨模态理解。

* **Understanding(理解)**
* **Perception(感知)**
* **Reasoning(推理)**
* **Decision-Making(决策)**

## Dataset Contents

- **train**
- * `train_qa.jsonl`（94.5 MB）：9267 个训练问答对
- *  signal (108 MB) ：包含 500 个时间步长×2 个传感器通道的训练时间序列数据

- **test**
- * `test_qa.jsonl`（23.6 MB）：2318 个测试问答对
- *  signal (20.8 MB)：包含 500 个时间步长×2 个传感器通道的测试时间序列数据

Total: 横跨四个任务的11585个QA实例。

## Tasks and Annotations


每个问答样本都与特定的时间序列段对齐（ID 与 `Singal` 中的 `data_ID` 匹配），并标注了以下内容：

| Task            | Input Scope  | Format          | Evaluation Metrics |
| --------------- | ------------ | --------------- | ------------------ |
| Understanding   | Single-cycle | Open-ended      | BLEU, ROUGE-L      |
| Perception      | Single-cycle | Multiple-choice | Accuracy, F1       |
| Reasoning       | Multi-cycle  | Multiple-choice | Accuracy, F1       |
| Decision-Making | Multi-cycle  | Open-ended      | BLEU, ROUGE-L      |

每个“对话”条目包含多回合问答，注释如下：

* `stage`: 任务类型 (1=Understanding, 2=Perception, 3=Reasoning, 4=Decision)
* `attribute`: 问题形式 (`open` or `close`)
* `value`: 自然语言提示 或 回答

## Example QA Entry
1 : Understanding
```json
{
    "id": [
      "1"
    ],
    "name": [
      "1.csv"
    ],
    "conversations": [
      {
        "from": "human",
        "stage": "1",
        "attribute": "open",
        "value": "请结合两相流理论，解读当前水平管中检测到的这段电压波动信号。\n <ts>"
      },
      {
        "from": "gpt",
        "value": "观察这段信号，高频脉动说明气相分布较为破碎。在气泡流下，这种电压走势直接对应了管内气相呈离散小气泡悬浮于液相，信号表现为高频小幅波动，代表相界面分布高度破碎且均匀。。由于水平管的约束，气泡在上升过程中会受到壁面剪切力的影响，导致波形呈现非对称特征。"
      }
    ]
  }
```
2 : Perception
```json
{
  "id": [
    "1"
  ],
  "name": [
    "1.csv"
  ],
  "conversations": [
    {
      "from": "human",
      "stage": "2",
      "attribute": "close",
      "value": "观察信号幅值，判断当前液流量区间。\n <ts>\na: 低液量(Ql<0.5) b: 中高液量(Ql>=0.5)"
    },
    {
      "from": "gpt",
      "value": "b"
    }
  ]
}
```
3 : Reasoning
```json
{
    "id": [
      "1"
    ],
    "name": [
      "1.csv"
    ],
    "conversations": [
      {
        "from": "human",
        "stage": "3",
        "attribute": "close",
        "value": "推断该系统在水平管工况下的演化趋势。\n <ts>\na: 趋于稳定 b: 维持现状 c: 存在失稳风险"
      },
      {
        "from": "gpt",
        "value": "b"
      }
    ]
  }
```
 4 : Decision
```json
{
    "id": [
      "1"
    ],
    "name": [
      "1.csv"
    ],
    "conversations": [
      {
        "from": "human",
        "stage": "4",
        "attribute": "open",
        "value": "当前混合速度Um为1.6420 m/s，为了降低管道压降并节能，应如何调整运行参数？\n <ts>"
      },
      {
        "from": "gpt",
        "value": "根据信号反馈的气泡流特征，当前运行点可能位于高耗能区。建议小幅降低气流速度，利用水平管的重力特性寻找流滑脱损失最小的临界点。若含气率(Yg)波动范围能缩小，则说明运行效率得到提升。"
      }
    ]
  }
```

## Applications

* 时间序列问答基准测试
* 引擎健康状况诊断
* 故障检测与预测
* 多模态模型训练与评估
* 跨模态语义对齐（文本与传感器信号）

## Construction Details

* 基于实验平台的管道流形数据集。
* 问题的设计遵循特定领域的规则，并由专家对大型语言模型（如 ChatGPT、DeepSeek）进行优化。
* 问答任务被划分以反映现实认知情况：

  * **理解**：信号意义的解读
  * **感知**：工况/流形识别
  * **推理**：演变趋势
  * **决策**：维护策略建议


