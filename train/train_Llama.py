import os
import argparse
import sys

# [Memory Optimization] Must be set before importing torch
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from torch.nn import CrossEntropyLoss
from peft import LoraConfig, get_peft_model, TaskType
import json
import math
import random
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Global Configuration Containers (Will be updated by argparse)
# ==============================================================================
# These act as placeholders so the Monkey Patches can access them globally
MODEL_PATH = ".../qwen2.5-coder-7B"

DATA_PATH = ".../data/qwen2.5-coder-7B/contrastive_keywords_add.jsonl"

OUTPUT_DIR = "./qwen2.5-coder-7B/attn_64_7"

TARGET_LAYERS = list(range(20, 28))
TARGET_HEADS = [0, 1]

ATTN_LOSS_SCALE_START = 40
ATTN_LOSS_SCALE_END = 0
ALPHA_MU = 1
ALPHA_TAU = 1.1
COMPLEXITY_SCALE = 2.0
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


CURRENT_GUIDANCE_COORDS = None
CURRENT_GUIDANCE_TARGETS = None
CURRENT_PAIR_ALPHAS = None
GLOBAL_LOSS_BUFFER = []


# ==============================================================================
# 2. Monkey Patch: Qwen3 Specific Attention (with QK-Norm)
# ==============================================================================
def manual_qwen3_attn_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor = None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    bsz, q_len, _ = hidden_states.size()

    if not hasattr(self, "head_dim"): self.head_dim = self.config.hidden_size // self.config.num_attention_heads

    hidden_shape = (*input_shape, -1, self.head_dim)

    q = self.q_proj(hidden_states).view(hidden_shape)
    k = self.k_proj(hidden_states).view(hidden_shape)
    v = self.v_proj(hidden_states).view(hidden_shape)

    # QK-Norm Logic
    if hasattr(self, "q_norm") and self.q_norm is not None:
        q = self.q_norm(q)
    if hasattr(self, "k_norm") and self.k_norm is not None:
        k = self.k_norm(k)

    query_states = q.transpose(1, 2)
    key_states = k.transpose(1, 2)
    value_states = v.transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

    if attention_mask is not None:
        if attention_mask.size() != attn_weights.size():
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        else:
            attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights


# ==============================================================================
# 3. Monkey Patch: Decoder Layer (Captures Loss)
# ==============================================================================
def custom_qwen3_decoder_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values=None,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    attn_outputs = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs
    )

    hidden_states_attn = attn_outputs[0]
    self_attn_weights = attn_outputs[1]

    hidden_states = residual + hidden_states_attn
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    # === Loss Calculation ===
    l_idx = getattr(self, "layer_idx", "Unknown")

    if (CURRENT_GUIDANCE_COORDS is not None and
            CURRENT_GUIDANCE_COORDS.size(0) > 0 and
            self_attn_weights is not None):

        # Note: TARGET_LAYERS and TARGET_HEADS are accessed from Global Scope
        if str(l_idx) != "Unknown" and int(l_idx) in TARGET_LAYERS:
            dev = hidden_states.device
            target_heads = getattr(self, "target_heads_idx", TARGET_HEADS)

            coords = CURRENT_GUIDANCE_COORDS.to(dev)
            targets = CURRENT_GUIDANCE_TARGETS.to(dev)

            # Use Pair Alphas if available
            if CURRENT_PAIR_ALPHAS is not None:
                alphas = CURRENT_PAIR_ALPHAS.to(dev)

            attn_permuted = self_attn_weights.permute(0, 2, 3, 1)

            if coords[:, 1].max() < attn_permuted.shape[1] and coords[:, 2].max() < attn_permuted.shape[2]:
                selected_values = attn_permuted[coords[:, 0], coords[:, 1], coords[:, 2]]
                target_head_values = selected_values[:, target_heads]
                targets_expanded = targets.unsqueeze(1).expand_as(target_head_values)

                mse = (target_head_values - targets_expanded) ** 2
                weighted_mse = mse
                layer_loss = weighted_mse.mean()

                if layer_loss.requires_grad or layer_loss.item() > 1e-9:
                    GLOBAL_LOSS_BUFFER.append(layer_loss)

    del self_attn_weights
    return hidden_states


# ==============================================================================
# 4. Patch: RMSNorm
# ==============================================================================
def custom_norm_forward(self, hidden_states):
    while isinstance(hidden_states, tuple): hidden_states = hidden_states[0]
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


def apply_qwen3_patch(model):
    print("\n[Patching] Applying Qwen3 Specific Patches...")
    try:
        layers = model.model.layers
        DecoderLayerClass = type(layers[0])
        AttentionClass = type(layers[0].self_attn)
        NormClass = type(layers[0].input_layernorm)

        AttentionClass.forward = manual_qwen3_attn_forward
        DecoderLayerClass.forward = custom_qwen3_decoder_forward
        NormClass.forward = custom_norm_forward

        print("✅ Patches Applied.")
    except Exception as e:
        print(f"❌ Patching failed: {e}")
        exit(1)


# ==============================================================================
# 6. Dataset & Collator
# ==============================================================================
class AttentionGuidedDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, tokenizer):
        self.data = data_list
        self.tokenizer = tokenizer

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return self.data[idx]


class AttentionGuidedCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, batch):
        batch_input_ids, batch_labels, batch_metrics = [], [], []
        guidance_coords, guidance_targets = [], []

        for batch_idx, item in enumerate(batch):
            messages = item.get('messages', [])
            base_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            full_text = base_prompt
            segments = item.get('meta_info', {}).get('segments', [])

            for seg in segments:
                full_text += f"{seg['prefix']}\n{seg['content']}\n"
            full_text += self.tokenizer.eos_token

            tokens = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
            input_ids = tokens.input_ids[0]
            seq_len = len(input_ids)

            labels = input_ids.clone()
            metrics_tensor = torch.zeros((seq_len, 2), dtype=torch.float32)

            base_tokens_len = len(self.tokenizer(base_prompt, add_special_tokens=True).input_ids)
            labels[:base_tokens_len] = -100
            current_offset = base_tokens_len
            temp_text = base_prompt

            for seg in segments:
                seg_content = f"{seg['prefix']}\n{seg['content']}\n"
                temp_text += seg_content
                new_len = len(self.tokenizer(temp_text, add_special_tokens=True).input_ids)

                start = current_offset
                end = min(new_len, seq_len)

                if start < end:
                    if not seg.get('trainable', False):
                        labels[start:end] = -100

                    comp = seg.get('complexity', 0.0)
                    uncert = seg.get('uncertainty', seg.get('perplexity', 0.0))
                    metrics_tensor[start:end, :] = torch.tensor([comp, uncert])

                    if seg.get('trainable', False) and 'key_attention_indices' in seg:
                        indices = seg['key_attention_indices']
                        if indices:
                            target_val = 1
                            for q_idx in range(start, end):
                                valid_keys = [k for k in indices if k < q_idx]
                                if valid_keys:
                                    local_target = 1
                                    for k_idx in valid_keys:
                                        guidance_coords.append([batch_idx, q_idx, k_idx])
                                        guidance_targets.append(local_target)

                current_offset = end

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_metrics.append(metrics_tensor)

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True,
                                                           padding_value=self.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)
        padded_metrics = torch.nn.utils.rnn.pad_sequence(batch_metrics, batch_first=True, padding_value=0.0)

        if len(guidance_coords) > 0:
            guidance_coords = torch.tensor(guidance_coords, dtype=torch.long)
            guidance_targets = torch.tensor(guidance_targets, dtype=torch.float32)
        else:
            guidance_coords = torch.empty((0, 3), dtype=torch.long)
            guidance_targets = torch.empty((0,), dtype=torch.float32)

        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "metrics": padded_metrics,
            "guidance_coords": guidance_coords,
            "guidance_targets": guidance_targets
        }


# ==============================================================================
# 7. Trainer
# ==============================================================================
class AttentionGuidedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_lm_loss = None
        self.last_attn_loss = None
        self.log_file_path = os.path.join(self.args.output_dir, "training_logs.jsonl")
        os.makedirs(self.args.output_dir, exist_ok=True)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        guidance_coords = inputs.pop("guidance_coords", None)
        guidance_targets = inputs.pop("guidance_targets", None)
        metrics = inputs.pop("metrics", None)
        labels = inputs.get("labels")

        global CURRENT_GUIDANCE_COORDS, CURRENT_GUIDANCE_TARGETS, CURRENT_PAIR_ALPHAS
        global GLOBAL_LOSS_BUFFER

        GLOBAL_LOSS_BUFFER.clear()

        if guidance_coords is not None and guidance_coords.size(0) > 0:
            dev = model.device
            # COMPLEXITY_SCALE and ALPHA params are accessed from Global Scope
            raw_score = COMPLEXITY_SCALE * metrics[..., 0] + metrics[..., 1]
            alpha_map = torch.sigmoid((raw_score - ALPHA_MU) / ALPHA_TAU).to(dev)

            guidance_coords = guidance_coords.to(dev)
            guidance_targets = guidance_targets.to(dev)
            pair_alphas = alpha_map[guidance_coords[:, 0], guidance_coords[:, 1]]

            CURRENT_GUIDANCE_COORDS = guidance_coords
            CURRENT_GUIDANCE_TARGETS = guidance_targets
            CURRENT_PAIR_ALPHAS = pair_alphas
        else:
            CURRENT_GUIDANCE_COORDS = None

        outputs = model(**inputs, output_attentions=False)

        att_loss = torch.tensor(0.0, device=model.device)
        if len(GLOBAL_LOSS_BUFFER) > 0:
            valid_losses = [l for l in GLOBAL_LOSS_BUFFER if l.item() > 1e-9]
            if valid_losses:
                att_loss = torch.stack(valid_losses).mean()

        lm_loss = outputs.loss
        if lm_loss is None:
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = CrossEntropyLoss()(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))

        # ATTN_LOSS_SCALE params accessed from Global Scope
        current_scale = ATTN_LOSS_SCALE_START + (ATTN_LOSS_SCALE_END - ATTN_LOSS_SCALE_START) * (
            min(1.0, self.state.global_step / max(1, self.state.max_steps)))
        total_loss = lm_loss + current_scale * att_loss

        if self.model.training:
            self.last_lm_loss = lm_loss.item()
            self.last_attn_loss = att_loss.item()
            if self.state.global_step % 5 == 0:
                print(f"Step {self.state.global_step}: LM={self.last_lm_loss:.4f}, Attn={self.last_attn_loss:.6f}")

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs, *args, **kwargs):
        if self.last_lm_loss is not None: logs["lm_loss"] = self.last_lm_loss
        if self.last_attn_loss is not None: logs["attn_loss"] = self.last_attn_loss
        super().log(logs, *args, **kwargs)
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(logs, ensure_ascii=False) + "\n")
        except:
            pass


def plot_loss_curves(log_history, output_dir):
    steps, lm_losses, attn_losses = [], [], []
    for entry in log_history:
        if "step" in entry and "lm_loss" in entry and "attn_loss" in entry:
            steps.append(entry["step"])
            lm_losses.append(entry["lm_loss"])
            attn_losses.append(entry["attn_loss"])

    if not steps: return
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(steps, lm_losses, label="LM Loss", color='blue')
    plt.title("LM Loss");
    plt.grid(True);
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(steps, attn_losses, label="Attn Loss", color='orange')
    plt.title("Attn Loss");
    plt.grid(True);
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"))


# ==============================================================================
# 8. Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="LoRA Training with Attention Guidance")

    # Paths
    parser.add_argument("--model_path", type=str, required=True, help="Base model path")
    parser.add_argument("--data_path", type=str, required=True, help="Training data jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    # Training Params
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--epochs", type=int, default=2)

    # Attention Guidance Params
    parser.add_argument("--target_layers_start", type=int, default=20, help="Start layer index (inclusive)")
    parser.add_argument("--target_layers_end", type=int, default=28, help="End layer index (exclusive)")
    parser.add_argument("--target_heads", type=str, default="0,1", help="Comma separated head indices")

    parser.add_argument("--attn_loss_start", type=float, default=40.0)
    parser.add_argument("--attn_loss_end", type=float, default=0.0)
    parser.add_argument("--alpha_mu", type=float, default=-10.0)
    parser.add_argument("--alpha_tau", type=float, default=1.1)
    parser.add_argument("--complexity_scale", type=float, default=1.0)

    # LoRA Params
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    args = parser.parse_args()

    # --- UPDATE GLOBALS FROM ARGS ---
    global MODEL_PATH, DATA_PATH, OUTPUT_DIR, TARGET_LAYERS, TARGET_HEADS
    global ATTN_LOSS_SCALE_START, ATTN_LOSS_SCALE_END, ALPHA_MU, ALPHA_TAU, COMPLEXITY_SCALE

    MODEL_PATH = args.model_path
    DATA_PATH = args.data_path
    OUTPUT_DIR = args.output_dir
    TARGET_LAYERS = list(range(args.target_layers_start, args.target_layers_end))
    TARGET_HEADS = [int(x) for x in args.target_heads.split(",")]
    ATTN_LOSS_SCALE_START = args.attn_loss_start
    ATTN_LOSS_SCALE_END = args.attn_loss_end
    ALPHA_MU = args.alpha_mu
    ALPHA_TAU = args.alpha_tau
    COMPLEXITY_SCALE = args.complexity_scale

    lora_modules_list = args.lora_modules.split(",")

    print(f"--- Configuration ---")
    print(f"Model: {MODEL_PATH}")
    print(f"Layers: {TARGET_LAYERS}")
    print(f"Heads: {TARGET_HEADS}")
    print(f"Attn Loss: {ATTN_LOSS_SCALE_START} -> {ATTN_LOSS_SCALE_END}")
    print(f"---------------------")

    # --- Load Model ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print("\n[Model] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model.resize_token_embeddings(len(tokenizer))
    model.enable_input_require_grads()
    model.config.use_cache = False

    apply_qwen3_patch(model)

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=lora_modules_list
    ))
    model.print_trainable_parameters()

    # Tagging Layers
    def find_layers_list(module):
        if hasattr(module, "layers") and isinstance(module.layers, torch.nn.ModuleList): return module.layers
        for name in ["base_model", "model", "transformer"]:
            if hasattr(module, name):
                found = find_layers_list(getattr(module, name))
                if found is not None: return found
        return None

    layers_list = find_layers_list(model)
    if layers_list:
        for idx, layer in enumerate(layers_list):
            layer.layer_idx = idx
            layer.target_heads_idx = TARGET_HEADS

    # Data
    all_data = []
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'r') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                    if 'id' not in item: item['id'] = f"idx_{i}"
                    all_data.append(item)
                except:
                    continue

    random.seed(42)
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.998)
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]

    train_dataset = AttentionGuidedDataset(train_data, tokenizer)
    eval_dataset = AttentionGuidedDataset(eval_data, tokenizer)
    collator = AttentionGuidedCollator(tokenizer)

    trainer = AttentionGuidedTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            logging_steps=32,
            eval_strategy="steps", eval_steps=50,
            save_strategy="steps", save_steps=100,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
            report_to="none"
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator
    )

    print("Starting Training...")
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    plot_loss_curves(trainer.state.log_history, OUTPUT_DIR)


if __name__ == "__main__":
    main()