import json
import os
import torch
import torch_npu.npu as npu
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    AutoModelForCausalLM,
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
    data_path   = "./data_class_5/labeled_logs_with_context_duplicated.csv"
    output_dir  = "./output_lora_class_5"
    os.makedirs(output_dir, exist_ok=True)
    model_path  = "./model/LLM-Research/Meta-Llama-3___1-8B-Instruct/"

    # -------------------------
    # 设备选择（支持 NPU）
    # -------------------------
    if npu.is_available():
        try:
            npu.set_device(0)
        except Exception:
            print("警告：npu.set_device(0) 调用失败，继续使用 'npu:0'")
        device = torch.device("npu:0")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    # ========== 1. 加载 tokenizer ==========
    merged_dir = "./merged_with_causal_lora"
    tokenizer = AutoTokenizer.from_pretrained(merged_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========== 2. 初始化 SequenceClassification 模型 ==========
    print("初始化 SequenceClassification 模型...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        merged_dir,
        num_labels=5,
        torch_dtype=torch.bfloat16 if device.type == 'npu' else None,
        trust_remote_code=True if "Meta-Llama" in model_path else False,
    )

    # id2label 和 label2id 映射
    id2label = {0: "DEBUG", 1: "INFO", 2: "WARN", 3: "ERROR", 4: "FATAL"}
    label2id = {v: k for k, v in id2label.items()}
    base_model.config.id2label = id2label
    base_model.config.label2id = label2id

    # ========== 3. 构建新的 SEQ_CLS LoRA ==========
    lora_cfg = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False,
                          r=16, lora_alpha=32, lora_dropout=0.05)
    model = get_peft_model(base_model, lora_cfg)
    model.enable_input_require_grads()

    model.config.pad_token_id = tokenizer.pad_token_id
    try:
        model.base_model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # ========== 4. 数据集处理 ==========
    raw_datasets = load_dataset("csv", data_files={"full": data_path})
    dataset      = raw_datasets["full"].map(lambda x: {"label": int(x["label"])})
    tokenized    = dataset.map(
        lambda ex: preprocess_function(ex, tokenizer),
        batched=True,
        num_proc=8,
        remove_columns=[c for c in dataset.column_names if c != "label"],
    )
    splits       = tokenized.train_test_split(test_size=0.2, seed=42)
    train_ds     = splits["train"]
    eval_ds      = splits["test"]

    # ========== 5. 训练配置 ==========
    args = TrainingArguments(
        output_dir="./chatbot_class_5",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=8,
        logging_steps=10,
        num_train_epochs=40,
        gradient_checkpointing=True,
        optim="adamw_torch",
        label_names=["labels"],
        eval_strategy="epoch",
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
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
    )

    # ========== 6. 训练 ==========
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ========== 7. 评估 ==========
    eval_results = trainer.evaluate()
    print("\n最终评估：", eval_results)


if __name__ == "__main__":
    main()
