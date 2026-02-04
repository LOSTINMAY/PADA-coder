import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# ================= 配置区域 =================
# --- 1. 配置参数 ---
model_id = "/root/autodl-tmp/attentioncode/qwen2.5-coder-7B"
data_file = "/root/autodl-tmp/attentioncode/data/qwen2.5-coder-7B/mixed_training_data.jsonl"
output_dir = "./qwen2.5-coder-7B/qwen_lora_output1"

# 训练参数
MAX_LENGTH = 8192
BATCH_SIZE = 1
GRAD_ACCUM = 8  # 建议增加累积步数，混合任务下大 Batch 更稳
NUM_EPOCHS = 1
LEARNING_RATE = 1e-5

# [重要] 恢复特殊 Token 定义
special_tokens = [
    "[GEN_GLOBAL_PLAN]", "[Algorithm]", "[GEN_PLAN]", "[GEN_CODE]",
    "[Record]", "[Record analysis]", "[PLAN_VERIFICATION]", "[Results Compare]"
    "[START_PLAN]","[END_PLAN]","[START_PROBLEM]","[END_PROBLEM]"
]


# ===========================================

def main():
    # --- 1. 加载 Tokenizer ---
    print(f"正在加载 Tokenizer: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # [修复1] 添加特殊 Token
    num_added_toks = tokenizer.add_tokens(special_tokens)
    print(f"已添加 {num_added_toks} 个特殊 Token")

    # --- 2. 核心：多轮对话数据处理函数 ---
    def process_func(example):
        input_ids = []
        labels = []
        for i, msg in enumerate(example['messages']):
            if i == 0:
                prev_ids = []
            else:
                prev_ids = tokenizer.apply_chat_template(
                    example['messages'][:i],
                    tokenize=True,
                    add_generation_prompt=False
                )
            curr_ids = tokenizer.apply_chat_template(
                example['messages'][:i + 1],
                tokenize=True,
                add_generation_prompt=False
            )
            new_token_ids = curr_ids[len(prev_ids):]
            input_ids.extend(new_token_ids)
            if msg['role'] == 'assistant':
                labels.extend(new_token_ids)
            else:
                labels.extend([-100] * len(new_token_ids))

        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        }

    # --- 3. 加载并处理数据 ---
    print("正在加载并处理数据...")
    dataset = load_dataset("json", data_files=data_file, split="train")

    # [可选] 如果你需要之前提到的“自动添加停止符”功能，请在这里插入 dataset.map(add_eos_token_batch)

    tokenized_dataset = dataset.map(process_func, remove_columns=dataset.column_names, num_proc=4)
    print(f"数据处理完成，共 {len(tokenized_dataset)} 条样本")

    # --- 4. 加载模型 ---
    print("正在加载模型 (BFloat16 + SDPA)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa"
    )

    # [修复2] 调整 Embedding 大小
    model.resize_token_embeddings(len(tokenizer))

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # --- 5. 配置 LoRA ---
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        # [建议] 增加 MLP 层 (gate/up/down) 效果通常更好
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # [修复3] 必须保存 embed_tokens，因为我们加了新词！
        modules_to_save=["embed_tokens", "lm_head"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- 6. 训练参数 ---
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=5,
        save_total_limit=1,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        group_by_length=True,
        dataloader_num_workers=2,
        report_to="none"
    )

    # --- 7. 开始训练 ---
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    print("🚀 开始训练...")
    trainer.train()

    # --- 8. 保存模型 ---
    print(f"✅ 训练完成，正在保存到 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()