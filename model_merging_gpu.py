import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def merge_lora_to_base_model(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
):
    """
    将 LoRA (CAUSAL_LM) 权重合并到基座模型，并保存到 output_path
    """
    os.makedirs(output_path, exist_ok=True)

    # 1. 加载 tokenizer（必须与 LoRA 训练时一致）
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载基座 CausalLM 模型
    print("加载基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True if "Meta-Llama" in base_model_path else False,
    )

    # 保持词表大小一致
    base_model.resize_token_embeddings(len(tokenizer))

    # 3. 加载 LoRA 权重
    print("加载 LoRA 权重...")
    peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    # 4. 合并 LoRA 到基座权重
    print("合并 LoRA 权重到基座模型...")
    merged_model = peft_model.merge_and_unload()

    # 5. 保存合并后的模型和 tokenizer
    print(f"保存合并后的模型到 {output_path} ...")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("✅ 合并完成！")

if __name__ == "__main__":
    # ==== 配置路径 ====
    BASE_MODEL_PATH = "./model/LLM-Research/Meta-Llama-3___1-8B-Instruct/"
    LORA_ADAPTER_PATH = "./output_lora_pretrain"  # LoRA 权重目录
    OUTPUT_PATH = "./merged_with_causal_lora"     # 合并后保存目录

    merge_lora_to_base_model(BASE_MODEL_PATH, LORA_ADAPTER_PATH, OUTPUT_PATH)
