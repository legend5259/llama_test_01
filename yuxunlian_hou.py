import json
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding, EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
import numpy as np

# -------------------------
# 数据预处理（分类任务）
# -------------------------
def preprocess_function(examples, tokenizer):
    texts = []
    for p, c, n in zip(examples["prev_log"], examples["curr_log"], examples["next_log"]):
        p = p or ""
        n = n or ""
        texts.append(f"{tokenizer.cls_token} {p} {tokenizer.sep_token} {c} {tokenizer.sep_token} {n}")
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
# 评价指标
# -------------------------
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# -------------------------
# 主程序
# -------------------------
def main():
    # 路径配置
    data_path   = "./data/labeled_logs_with_context.csv"
    output_dir  = "./output"
    os.makedirs(output_dir, exist_ok=True)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path  = "./model/LLM-Research/Meta-Llama-3___1-8B-Instruct/"

    # ========== 1. 加载基座模型并合并已有 CAUSAL_LM LoRA ==========
    adapter_dir = "./output_lora_pretrain"    # 已训练的 CAUSAL_LM LoRA 路径
    merged_dir  = "./merged_with_causal_lora" # 合并后的模型保存路径

    # 1. 加载 tokenizer（必须用 LoRA 训练时的）
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    os.makedirs(merged_dir, exist_ok=True)

    print("加载基座 CausalLM...")
    base_causal = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True if "Meta-Llama" in model_path else False,
    )
    # 关键：保持和 LoRA 训练时相同的词表大小
    base_causal.resize_token_embeddings(len(tokenizer))

    print("加载 CAUSAL_LM LoRA...")
    peft_causal = PeftModel.from_pretrained(base_causal, adapter_dir)

    print("合并 LoRA 到基座权重...")
    merged_model = peft_causal.merge_and_unload()

    print(f"保存合并后的模型到 {merged_dir} ...")
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    # ========== 2. 用合并后的权重初始化分类模型 ==========
    print("初始化 SequenceClassification 模型...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        merged_dir,
        num_labels=2,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True if "Meta-Llama" in model_path else False,
    )

    # ========== 3. 构建新的 SEQ_CLS LoRA ==========
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05
    )
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
        output_dir="./chatbot",
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

    # ========== 7. 阈值搜索 ==========
    eval_results = trainer.evaluate()
    print("\n最终评估：", eval_results)

    preds_output = trainer.predict(eval_ds)
    probs = torch.softmax(torch.tensor(preds_output.predictions), dim=-1).numpy()
    anomaly_scores = probs[:, 1]
    true_labels    = preds_output.label_ids

    precisions, recalls, thresholds = precision_recall_curve(true_labels, anomaly_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx  = np.nanargmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1     = f1_scores[best_idx]

    print(f"验证集最优 F1 = {best_f1:.4f} 对应阈值 = {best_thresh:.4f}")
    threshold_info = {"best_threshold": float(best_thresh), "best_f1": float(best_f1)}
    os.makedirs("./yuzhi", exist_ok=True)
    with open("./yuzhi/best_threshold.json", "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, ensure_ascii=False, indent=2)

    print("最佳阈值已保存到 ./yuzhi/best_threshold.json")

if __name__ == "__main__":
    main()
