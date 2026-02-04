import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
import os

# ================= 配置路径 =================
# 1. 基础模型路径
BASE_MODEL_PATH = "/root/autodl-tmp/attentioncode/qwen2.5-coder-7B"

# 2. LoRA 权重路径
LORA_PATH = "/root/autodl-tmp/attentioncode/train/qwen2.5-coder-7B/qwen_lora_output/checkpoint-52"

# 3. 合并后的模型保存路径 (任意你想要的新位置)
MERGED_MODEL_PATH = "/root/autodl-tmp/attentioncode/merge_model/qwen2.5-coder-7B_lora"


# ===========================================

def merge_lora():
    print(f"Loading tokenizer from {LORA_PATH}...")
    # 1. 先拿到实际的 Tokenizer 长度 (例如 151678)
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    actual_len = len(tokenizer)
    print(f"Actual tokenizer length: {actual_len}")

    print(f"Loading base model from {BASE_MODEL_PATH}...")
    # 2. 加载 Base Model (先不要修改 Config 的 vocab_size)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 3. 【关键步骤 A】先 Resize 到 LoRA 训练时的大小 (151678)
    # 这样 PeftModel 才能加载进去，否则会报 size mismatch
    print(f"Step A: Resizing to match LoRA checkpoint size ({actual_len})...")
    base_model.resize_token_embeddings(actual_len)

    print(f"Loading LoRA adapters from {LORA_PATH}...")
    # 4. 加载 LoRA
    model = PeftModel.from_pretrained(base_model, LORA_PATH)

    print("Merging weights...")
    # 5. 合并权重
    model = model.merge_and_unload()

    '''# 6. 【关键步骤 B】合并完成后，再 Resize 到 vLLM 需要的对齐大小 (151936)
    print(f"Step B: Resizing to TARGET size for vLLM ({TARGET_VOCAB_SIZE})...")
    model.resize_token_embeddings(TARGET_VOCAB_SIZE)
    print(f"Current embedding shape: {model.get_input_embeddings().weight.shape}")

    # 7. 强制修改 Config，确保 vLLM 看到正确的数字
    model.config.vocab_size = TARGET_VOCAB_SIZE'''

    print(f"Saving merged model to {MERGED_MODEL_PATH}...")
    model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)

    print("Done! Model merged and padded successfully.")


if __name__ == "__main__":
    merge_lora()