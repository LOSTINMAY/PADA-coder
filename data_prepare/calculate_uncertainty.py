import argparse
import json
import torch
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_local_model(model_path, device_map="auto"):
    print(f"Loading model from: {model_path} ...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True
    )
    model.eval()
    return tokenizer, model


def get_token_intervals(tokenizer, full_text, segments):
    """
    Search the content of segments sequentially in full_text and map it to the token index.
    """
    encodings = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encodings.input_ids
    offset_mapping = encodings.offset_mapping

    token_intervals = []
    current_char_ptr = 0

    for seg in segments:
        content = seg['content']
        start_char = full_text.find(content, current_char_ptr)

        if start_char == -1:
            token_intervals.append(None)
            continue

        end_char = start_char + len(content)
        current_char_ptr = end_char

        seg_token_start = None
        seg_token_end = None

        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_end > start_char and tok_start < end_char:
                if seg_token_start is None:
                    seg_token_start = i
                seg_token_end = i + 1

        if seg_token_start is not None and seg_token_end is not None:
            token_intervals.append((seg_token_start, seg_token_end))
        else:
            token_intervals.append(None)

    return input_ids, token_intervals


def calculate_metrics(model, input_ids_tensor, intervals):
    with torch.no_grad():
        outputs = model(input_ids=input_ids_tensor)
        logits = outputs.logits  # [1, seq_len, vocab]
        log_probs = torch.log_softmax(logits, dim=-1)

    metrics_list = []

    for interval in intervals:
        if interval is None:
            metrics_list.append(None)
            continue

        start, end = interval
        pred_start = max(1, start)
        pred_end = end

        if pred_start >= pred_end:
            metrics_list.append(None)
            continue

        target_ids = input_ids_tensor[:, pred_start:pred_end]
        interval_log_probs = log_probs[:, pred_start - 1:pred_end - 1, :]

        token_log_probs = interval_log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        nll = -token_log_probs.mean().item()
        ppl = np.exp(nll)
        if ppl > 1e4: ppl = 1e4

        metrics_list.append({
            "uncertainty": round(nll, 4),
            "perplexity": round(ppl, 2)
        })

    return metrics_list


def main():
    parser = argparse.ArgumentParser(description="Calculate LLM Perplexity and Uncertainty")
    parser.add_argument("--model_path", type=str, required=True, help="Path to local model")
    parser.add_argument("--input_file", type=str, required=True, help="Input .jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Output .jsonl file")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map (auto, cuda, cpu)")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    tokenizer, model = load_local_model(args.model_path, args.device_map)
    device = model.device

    print(f"Processing: {args.input_file} -> {args.output_file}")

    with open(args.input_file, 'r', encoding='utf-8') as f_in, \
            open(args.output_file, 'w', encoding='utf-8') as f_out:

        lines = f_in.readlines()
        for line in tqdm(lines, desc="Processing Lines"):
            if not line.strip(): continue

            data = json.loads(line)

            # Filter logic (optional, dependent on your data schema)
            task_type = data.get('meta_info', {}).get('task_type', '')
            if 'coding' not in task_type and '_code' not in data.get('id', ''):
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            messages = data['messages']
            segments = data['meta_info']['segments']

            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            input_ids, intervals = get_token_intervals(tokenizer, full_text, segments)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

            metrics = calculate_metrics(model, input_tensor, intervals)

            for i, metric in enumerate(metrics):
                if i < len(segments) and metric is not None:
                    seg_type = segments[i].get('type')
                    # Apply only to specific segment types
                    if seg_type in ['local_plan', 'code']:
                        segments[i]['uncertainty'] = metric['uncertainty']
                        segments[i]['perplexity'] = metric['perplexity']

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Done! Metrics saved to {args.output_file}")


if __name__ == "__main__":
    main()