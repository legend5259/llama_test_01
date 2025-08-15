# train_lora_pretrain.py
import os
import json
import random
from functools import partial

import torch
import numpy as np
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
# from transformers import EarlyStoppingCallback  # 已移除（不再使用）

from peft import LoraConfig, get_peft_model, TaskType

# ---------- hyperparams / paths ----------
data_path   = "./data_class_5/labeled_logs.csv"  # CSV with columns: label,log_text (label can be ignored)
model_path  = "./model/LLM-Research/Meta-Llama-3___1-8B-Instruct/"  # base model dir
output_dir  = "./output_lora_pretrain"
os.makedirs(output_dir, exist_ok=True)

max_length      = 512
mask_prob       = 0.15         # 掩码比例（按 token 计）
prompt_template = "Please restore the masked parts in the following log text using the provided context:\n"
seed            = 42
# ----------------------------------------

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def add_mask_token_if_needed(tokenizer):
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    return tokenizer

def random_token_mask(token_ids, mask_token_id, mask_prob):
    n = len(token_ids)
    if n == 0:
        return token_ids, []
    masked = token_ids.copy()
    mask_positions = []
    for i in range(n):
        if random.random() < mask_prob:
            masked[i] = mask_token_id
            mask_positions.append(i)
    return masked, mask_positions

def preprocess_batch(examples, tokenizer, prompt_template, max_length, mask_prob):
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    for txt in examples["log_text"]:
        if txt is None:
            txt = ""
        full_encoded = tokenizer(
            txt,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None,
        )["input_ids"]
        masked_ids, _ = random_token_mask(full_encoded, tokenizer.mask_token_id, mask_prob)

        prompt_enc = tokenizer(
            prompt_template,
            add_special_tokens=True,
            truncation=True,
            max_length=64,
        )["input_ids"]

        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        seq = prompt_enc + masked_ids + eos + full_encoded

        labels = [-100] * (len(prompt_enc) + len(masked_ids) + len(eos)) + full_encoded.copy()

        if len(seq) > max_length:
            seq = seq[-max_length:]
            labels = labels[-max_length:]

        pad_len = max_length - len(seq)
        input_ids = seq + [tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * len(seq) + [0] * pad_len
        labels = labels + [-100] * pad_len

        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        labels_batch.append(labels)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels": labels_batch,
    }

def main():
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer = add_mask_token_if_needed(tokenizer)

    # load dataset csv (assume columns label,log_text)
    raw = load_dataset("csv", data_files={"full": data_path})
    ds = raw["full"]

    # 不再划分，直接把整个数据集当作训练集
    train_ds = ds

    # load model as causal LM
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True if "Meta-Llama" in model_path else False,
    )

    base_model.resize_token_embeddings(len(tokenizer))

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.enable_input_require_grads()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preprocess_fn = partial(
        preprocess_batch,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        max_length=max_length,
        mask_prob=mask_prob,
    )

    # 只对训练集做 map，不保留 eval 集
    train_tokenized = train_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=[c for c in train_ds.column_names if c not in ("log_text", "label")],
    )

    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Training args：只训练 1 轮，关闭评估
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        logging_steps=20,
        eval_strategy="no",      # 关闭评估
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="adamw_torch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        tokenizer=tokenizer,
        # 不使用早停回调（已移除）
        data_collator=lambda data: {
            "input_ids": torch.stack([d["input_ids"] for d in data]),
            "attention_mask": torch.stack([d["attention_mask"] for d in data]),
            "labels": torch.stack([d["labels"] for d in data]),
        },
    )

    # train
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("训练完成，结果保存在：", output_dir)

if __name__ == "__main__":
    main()
