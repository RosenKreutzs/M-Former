#!/usr/bin/env python
# -*- coding:utf-8 _*-
import importlib
from transformers import Trainer, TrainingArguments
import torch



#对 Hugging Face Trainer训练器的二次封装
class Exp_Pretrain(Trainer):
    def __init__(self, args, train_dataset,data_collator=None, eval_dataset=None):
        # 构建模型
        model = self._build_model(args)

        # 将 args 中的参数（如学习率、Batch Size）打包成官方标准格式。
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            logging_dir=args.output_dir,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_strategy="steps" if eval_dataset else "no",
            save_total_limit=2,
            num_train_epochs=args.num_train_epochs,
            report_to=args.report_to,  # Example: Integrate TensorBoard
            remove_unused_columns=False,
        )

        #调用父类的初始化方法，把刚才准备好的模型、数据和配置全部交给 Trainer 接管。
        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics if eval_dataset else None,
        )

    def _build_model(self, args):
        """根据配置动态加载模型。"""
        module = importlib.import_module("models." + args.model)
        model = module.Model(
            args
        ).cuda()
        return model
    

    








