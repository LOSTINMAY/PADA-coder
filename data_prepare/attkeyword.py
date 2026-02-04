import json
import torch
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc
import os
import shutil
import string
import argparse
# --- 1. configure ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# --- 2. separator ---
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to", "for",
    "and", "or", "but", "so", "if", "then", "else", "when", "while", "return", "def",
    "class", "self", "import", "from", "as", "pass", "none", "true", "false",
    "print", "assert", "main", "range", "len",
    "\n", "\t", " ", "","_",",","`",":"
}


class SegmentedAttentionExtractor:
    def __init__(self, model_path):
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        '''# 2. define new Token
        new_tokens = [
            "[GEN_GLOBAL_PLAN]", "[Algorithm]", "[GEN_PLAN]",
            "[GEN_CODE]", "[PLAN_VERIFICATION]", "[Record analysis]",
            "[Record]", "[Results Compare]", "[START_PROBLEM]",
            "[END_PROBLEM]", "[START_PLAN]", "[END_PLAN]"
        ]

        # 3. Add to Tokenizer
        num_added_toks = self.tokenizer.add_tokens(new_tokens, special_tokens=False)

        print(f"Added {num_added_toks} tokens")'''

        max_memory_mapping = {0: "12GiB", "cpu": "128GiB"}
        offload_dir = "./model_offload_seg_temp"
        if not os.path.exists(offload_dir): os.makedirs(offload_dir)
        '''config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.vocab_size = 151936'''
        print(len(self.tokenizer))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # config = config,
            device_map="auto",
            max_memory=max_memory_mapping,
            offload_folder=offload_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager"
        )


        self.model.config.output_attentions = True
        self.device = "cuda"

        # Hook
        self.params = {
            "current_attn": None,
            "target_layer_start_idx": 20
        }
        self.hooks = []

    def _register_hooks(self):
        self.params["current_attn"] = None
        self.hooks = []

        def get_hook(layer_idx):
            def hook_fn(module, args, output):
                if not isinstance(output, tuple): return output
                if len(output) < 2 or output[1] is None: return output

                # output[1] shape: [Batch, Heads, Q_Len(Segment), K_Len(Total)]
                attn_weights = output[1]

                if layer_idx >= self.params["target_layer_start_idx"]:
                    # shape -> [Batch, Q_Len, K_Len]
                    layer_attn = torch.mean(attn_weights, dim=1).detach().to(torch.float32).cpu()

                    if self.params["current_attn"] is None:
                        self.params["current_attn"] = layer_attn
                    else:
                        self.params["current_attn"] += layer_attn

                new_output = (output[0], None) + output[2:]
                return new_output

            return hook_fn

        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, "self_attn"):
                handle = layer.self_attn.register_forward_hook(get_hook(i))
                self.hooks.append(handle)

    def _remove_hooks(self):
        for handle in self.hooks: handle.remove()
        self.hooks = []

    def is_valid_keyword(self, token_id):
        raw_token = self.tokenizer.decode([token_id])
        token_str = raw_token.strip()
        if not token_str: return False, token_str
        if len(token_str) == 1:
            if token_str in string.punctuation: return False, token_str
            if token_str.lower() not in ['i', 'j', 'k', 'n', 'm', 'x', 'y']: return False, token_str
        if token_str.startswith(":") or token_str.startswith("["): return False, token_str
        if all(char in string.punctuation for char in token_str): return False, token_str
        if token_str.lower() in STOP_WORDS: return False, token_str
        return True, token_str

    def extract_keywords(self, jsonl_file, output_file, top_k_ratio=0.05, max_score_threshold=100.0, window_size=1):
        processed_count = 0
        file_mode = 'w'

        if os.path.exists(output_file):
            print(f"Check existing output file: {output_file}")
            with open(output_file, 'r', encoding='utf-8') as f_check:
                for _ in f_check:
                    processed_count += 1

            if processed_count > 0:
                print(f"Found {processed_count} lines already processed. Resuming...")
                file_mode = 'a'
            else:
                print("Output file exists but is empty. Overwriting...")
        else:
            print("No existing output file. Starting fresh...")

        # ---------------------------

        with open(jsonl_file, 'r', encoding='utf-8') as fin, \
                open(output_file, file_mode, encoding='utf-8') as fout:

            for line_idx, line in enumerate(tqdm(fin, desc="Processing Lines")):

                if line_idx < processed_count:
                    continue

                try:
                    data = json.loads(line)
                except:
                    continue

                # --- 1.  Pre-construct a complete sequence of Token IDs ---
                # To ensure alignment, we first tokenize the entire text and then slice it to feed to the model
                if len(data['messages']) >= 2:
                    base_prompt = f"{data['messages'][0]['content']}\n{data['messages'][1]['content']}\n"
                else:
                    base_prompt = data['messages'][0]['content'] + "\n"

                segments = data['meta_info']['segments']
                full_text = base_prompt
                for seg in segments:
                    full_text += f"{seg['prefix']}\n{seg['content']}\n"

                # Full Tokenization
                full_inputs = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
                full_input_ids = full_inputs.input_ids  # CPU tensor
                all_token_ids_list = full_input_ids[0].tolist()

                # Calculate the start/end indices of each Segment in the full sequence
                # We need to go through the logic again to locate the index
                current_offset = len(self.tokenizer(base_prompt, add_special_tokens=True).input_ids)
                segment_metrics = []  # save (start, end, segment_obj)

                temp_text = base_prompt
                for seg in segments:
                    seg_text = f"{seg['prefix']}\n{seg['content']}\n"
                    temp_text += seg_text
                    new_len = len(self.tokenizer(temp_text, add_special_tokens=True).input_ids)

                    start = current_offset
                    end = new_len
                    end = min(end, full_input_ids.shape[1])

                    segment_metrics.append((start, end, seg))
                    current_offset = end

                # --- 2. KV Cache ---
                past_key_values = None
                processed_len = 0

                first_seg_start = segment_metrics[0][0]

                base_input_ids = full_input_ids[:, :first_seg_start].to(self.device)

                try:
                    with torch.no_grad():
                        # 1：Prefill Base Context
                        outputs = self.model(
                            base_input_ids,
                            use_cache=True,
                            output_attentions=False
                        )
                        past_key_values = outputs.past_key_values
                        processed_len = base_input_ids.shape[1]

                        del outputs, base_input_ids
                        torch.cuda.empty_cache()

                    for start, end, seg in segment_metrics:
                        if start >= end: continue

                        seg_input_ids = full_input_ids[:, start:end].to(self.device)
                        seg_len = seg_input_ids.shape[1]

                        # Constructing Attention Mask
                        # The shape should be [Batch, Total_Len] (Past + Current)
                        # However, transformers usually handle automatically, so we'll just pass in input_ids and past_key_values for now

                        # Only when trainable=True, do we enable the Hook to record Attention
                        capture_attention = seg.get('trainable', False)

                        if capture_attention:
                            self._register_hooks()

                        with torch.no_grad():
                            outputs = self.model(
                                seg_input_ids,
                                past_key_values=past_key_values,
                                use_cache=True,
                                output_attentions=True
                            )

                        # update KV Cache
                        past_key_values = outputs.past_key_values

                        # --- 3. find key tokens ---
                        if capture_attention and self.params["current_attn"] is not None:
                            # Accumulator: [1, Seg_Len, Total_History_Len]
                            avg_attn = self.params["current_attn"] / 8
                            attn_matrix = avg_attn.squeeze(0)  # [Seg_Len, Total_History_Len]

                            token_scores = torch.sum(attn_matrix, dim=0)

                            valid_len = token_scores.shape[0]

                            # --- Top-K & Window ---
                            dynamic_k = max(5, int(valid_len * top_k_ratio))
                            dynamic_k = min(50, dynamic_k)

                            _, top_indices = torch.topk(token_scores, min(dynamic_k * 3, valid_len))

                            candidate_indices = set()
                            for idx_tensor in top_indices:
                                idx = idx_tensor.item()
                                if idx == 0: continue
                                for w in range(-window_size, window_size + 1):
                                    n_idx = idx + w
                                    if 0 < n_idx < valid_len:
                                        candidate_indices.add(n_idx)

                            valid_indices = []
                            valid_values = []
                            for idx in sorted(list(candidate_indices)):
                                score = token_scores[idx].item()
                                if score > max_score_threshold: continue
                                is_valid, _ = self.is_valid_keyword(all_token_ids_list[idx])
                                if is_valid:
                                    valid_indices.append(idx)
                                    valid_values.append(round(score, 4))
                                if len(valid_indices) >= dynamic_k * 2: break

                            seg['key_attention_indices'] = valid_indices
                            seg['key_attention_values'] = valid_values

                        if capture_attention:
                            self._remove_hooks()
                            self.params["current_attn"] = None

                        processed_len += seg_len
                        del seg_input_ids, outputs
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM at line {line_idx}. Skipped.")
                        del past_key_values
                        self._remove_hooks()
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

                del past_key_values, full_input_ids
                gc.collect()
                torch.cuda.empty_cache()

        if os.path.exists("./model_offload_seg_temp"):
            shutil.rmtree("./model_offload_seg_temp")


def main():
    parser = argparse.ArgumentParser(description="Segmented Attention Keyword Extractor")

    # path
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--input_file", type=str, required=True, help="Input .jsonl file path")
    parser.add_argument("--output_file", type=str, required=True, help="Output .jsonl file path")

    parser.add_argument("--top_k_ratio", type=float, default=0.08, help="Ratio of top attention tokens to extract")
    parser.add_argument("--max_score_threshold", type=float, default=10.0,
                        help="Filter out tokens with scores above this")
    parser.add_argument("--window_size", type=int, default=1, help="Context window around high-attention tokens")
    parser.add_argument("--target_layer_start", type=int, default=20, help="Layer index to start collecting attention")

    # device
    parser.add_argument("--device_map", type=str, default="auto", help="Device map for transformers")

    args = parser.parse_args()

    extractor = SegmentedAttentionExtractor(args.model_path)

    extractor.params["target_layer_start_idx"] = args.target_layer_start

    extractor.extract_keywords(
        jsonl_file=args.input_file,
        output_file=args.output_file,
        top_k_ratio=args.top_k_ratio,
        max_score_threshold=args.max_score_threshold,
        window_size=args.window_size
    )


if __name__ == "__main__":
    main()
