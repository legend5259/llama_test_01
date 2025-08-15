import json
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback, AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# -------------------------
# 数据预处理（分类任务）
# -------------------------
def preprocess_function(examples, tokenizer):
    # 直接使用日志文本
    texts = examples["log_text"]
    result = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    if "label" in examples:
        result["labels"] = examples["label"]
    return result


# -------------------------
# 评价指标（多分类）
# -------------------------
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# -------------------------
# 主程序
# -------------------------
def main():
    # 路径配置
    data_path   = "../data_class_5/labeled_logs_selected.csv"
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path  = "../model/LLM-Research/Meta-Llama-3___1-8B-Instruct/"

    # ========== 1. 加载基座模型并合并已有 CAUSAL_LM LoRA ==========
    merged_dir  = "../merged_with_causal_lora"  # 合并后的模型保存路径

    # 1. 加载 tokenizer（必须用 LoRA 训练时的）
    tokenizer = AutoTokenizer.from_pretrained(merged_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # ========== 2. 用合并后的权重初始化分类模型 ==========
    print("初始化 SequenceClassification 模型...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        merged_dir,
        num_labels=5,  # 五分类
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True if "Meta-Llama" in model_path else False,
    )

    # id2label 和 label2id 映射，方便推理时查看类别名
    id2label = {
        0: "DEBUG",
        1: "INFO",
        2: "WARN",
        3: "ERROR",
        4: "FATAL",
    }
    label2id = {v: k for k, v in id2label.items()}
    base_model.config.id2label = id2label
    base_model.config.label2id = label2id

    # 3) 把 LoRA adapter 挂回去
    adapter_dir_lora = "../output_lora_class_5"
    model = PeftModel.from_pretrained(base_model, adapter_dir_lora)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    # ========== 4. 数据集处理 ==========
    raw_datasets = load_dataset("csv", data_files={"full": data_path})
    dataset      = raw_datasets["full"].map(lambda x: {"label": int(x["label"])})
    tokenized    = dataset.map(
        lambda ex: preprocess_function(ex, tokenizer),
        batched=True,
        num_proc=8,
        remove_columns=[c for c in dataset.column_names if c != "label"],
    )
    eval_ds = tokenized

    args = TrainingArguments(
        output_dir="../chatbot_class_5",
        per_device_train_batch_size=32,  # 训练 batch
        per_device_eval_batch_size=100,  # 评估/预测 batch
        gradient_accumulation_steps=8,
        logging_steps=10,
        num_train_epochs=40,
        gradient_checkpointing=True,
        optim="adamw_torch",
        label_names=["labels"],
        eval_strategy="epoch",  # 注意是 evaluation_strategy，不是 eval_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
    )

    # ── 在验证集上常规模型评估 ──
    eval_results = trainer.evaluate()
    print("\n最终评估：", eval_results)

    # --- 预测整个验证集，拿到预测结果 ---
    predictions_output = trainer.predict(eval_ds)
    preds_logits = predictions_output.predictions  # shape: (num_samples, num_labels)
    preds = preds_logits.argmax(axis=1)  # 取最大概率的类别标签（0~4）

    # 从原始 dataset 恢复 DataFrame
    raw_df = raw_datasets["full"].to_pandas()

    # 添加预测标签列
    raw_df["pred_label"] = preds
    # 如果需要显示标签名称，可以加上：
    id2label = {
        0: "DEBUG",
        1: "INFO",
        2: "WARN",
        3: "ERROR",
        4: "FATAL",
    }
    raw_df["pred_label_name"] = raw_df["pred_label"].map(id2label)

    # 保存到新文件
    output_pred_file = "predictions_with_label.csv"
    raw_df.to_csv(output_pred_file, index=False, encoding="utf-8")
    print(f"五分类预测标签已保存到：{output_pred_file}")


if __name__ == "__main__":
    main()
