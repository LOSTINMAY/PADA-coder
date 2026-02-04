import argparse
import json
import torch
import numpy as np
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch.nn.functional as F


def detect_value_spike(values, default_fallback=0.0):
    """
    Detect the 'elbow' or 'spike' in a sorted list of values (high to low).
    Simple implementation using the point of maximum curvature or relative drop.
    """
    if len(values) < 2:
        return values[0] if values else default_fallback

    # Sort descending
    sorted_vals = sorted(values, reverse=True)

    # Normalize
    max_val = sorted_vals[0]
    if max_val == 0: return 0.0
    norm_vals = [v / max_val for v in sorted_vals]

    # Find the point where the drop is most significant (simplified Kneedle algorithm idea)
    # We look for the index where the distance to the diagonal line is maximized
    n = len(norm_vals)
    max_dist = -1
    elbow_idx = 0

    for i in range(n):
        # Coord on curve: (i/n, norm_vals[i])
        # Coord on diagonal: (i/n, 1 - i/n)
        # Distance ~ (norm_vals[i] - (1-i/n))
        dist = norm_vals[i] - (1 - (i / n))
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i

    return sorted_vals[elbow_idx]


def next_stricter_spike(values, current_threshold):
    """
    Find a threshold strictly higher than the current one to reduce selection.
    """
    sorted_vals = sorted([v for v in values if v > current_threshold], reverse=True)
    if not sorted_vals:
        return current_threshold * 1.5  # Fallback: artificially raise

    # Find a new spike in the remaining upper subset
    return detect_value_spike(sorted_vals, default_fallback=current_threshold)

class KeywordConfirmer:
    def __init__(self, model_path, device_map="auto"):
        print(f"Loading Evaluator Model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"  # Use FlashAttn for speed if available
        )
        self.model.eval()
        self.device = self.model.device

    def calculate_ppl(self, context_ids, target_ids):
        """
        Calculate PPL of target_ids given context_ids.
        context_ids: [1, seq_len]
        target_ids: [1, target_len]
        """
        if target_ids.size(1) == 0: return 0.0

        input_ids = torch.cat([context_ids, target_ids], dim=1)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits  # [1, seq_len + target_len, vocab]

            # Shift logits so token n predicts n+1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Only calculate loss for the target part
            # Target starts at len(context_ids)
            target_start_idx = context_ids.size(1) - 1

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits[:, target_start_idx:, :].transpose(1, 2),
                            shift_labels[:, target_start_idx:])

            # PPL = exp(mean(loss))
            mean_loss = loss.mean()
            return torch.exp(mean_loss).item()

    def perturbation_analysis(self, base_prompt_ids, code_ids, tokens_to_perturb, original_ppl):
        """
        Compute Delta PPL for each token in tokens_to_perturb.
        perturbation = removing the token (ablation).
        """
        results = {}
        base_list = base_prompt_ids[0].tolist()

        # Optimization: Batch processing could be done here, but length varies after removal.
        # We process sequentially for safety with memory.

        for token_idx in tokens_to_perturb:
            # Create p_{\setminus t} (Remove token at token_idx)
            # Note: token_idx is relative to the full token list.
            # We assume base_prompt_ids contains the full plan text.

            if token_idx >= len(base_list): continue

            # Ablation: Slice out the token
            perturbed_list = base_list[:token_idx] + base_list[token_idx + 1:]
            perturbed_tensor = torch.tensor([perturbed_list], device=self.device)

            try:
                new_ppl = self.calculate_ppl(perturbed_tensor, code_ids)
                delta_ppl = new_ppl - original_ppl
                results[token_idx] = delta_ppl
            except Exception as e:
                print(f"Error computing PPL for token {token_idx}: {e}")
                results[token_idx] = 0.0

        return results

    def dynamic_filter(self,
                       v_s, v_t,
                       all_token_ids,
                       base_prompt_ids,
                       code_ids,
                       L,
                       k_ratio,
                       rho_min, rho_max):
        """
        Implementation of Algorithm 1: Dynamic select of k, eta, and tau.
        v_s: Student Attention Map {token_idx: score}
        v_t: Teacher Attention Map {token_idx: score}
        """

        # --- Phase 1: Initialization ---
        n_init = int(L * k_ratio)
        n_min = int(L * rho_min)
        n_max = int(L * rho_max)

        # Top-K selection (Indices)
        def get_top_k(score_map, n):
            sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
            return set([idx for idx, val in sorted_items[:n]])

        k_s = get_top_k(v_s, n_init)
        k_t = get_top_k(v_t, n_init)

        c_set = k_s.intersection(k_t)  # Consensus
        d_set = k_s.union(k_t) - c_set  # Disagreement

        # Baseline PPL (p)
        original_ppl = self.calculate_ppl(base_prompt_ids, code_ids)

        # --- Phase 2: Disagreement Verification ---
        # Analyze ALL tokens in D
        s_d = self.perturbation_analysis(base_prompt_ids, code_ids, list(d_set), original_ppl)

        # Detect tau (Init Logic Threshold)
        d_scores = list(s_d.values())
        tau = detect_value_spike(d_scores, default_fallback=0.1)

        d_sig = {t for t in d_set if s_d.get(t, 0) > tau}
        d_remain = d_set - d_sig

        # --- Phase 3: Iterative Density Adaptation ---
        # Init eta (Attention Threshold) from Consensus scores
        c_scores = [v_s.get(t, 0) + v_t.get(t, 0) for t in c_set]  # Sum or max
        eta = detect_value_spike(c_scores)

        k_final = set()

        # Loop limit to prevent infinite loops
        max_iter = 5
        iter_count = 0

        while iter_count < max_iter:
            iter_count += 1
            n_curr = len(c_set) + len(d_sig)

            # Case A: Overflow
            if n_curr > n_max:
                # Filter Consensus: Trust High-Attn, Verify Low-Attn
                # Low attn in C defined by eta
                c_low = {t for t in c_set if (v_s.get(t, 0) + v_t.get(t, 0)) / 2 < eta}

                # Verify c_low with Perturbation
                s_c = self.perturbation_analysis(base_prompt_ids, code_ids, list(c_low), original_ppl)
                c_verified = {t for t in c_low if s_c.get(t, 0) > tau}

                # Construct temp final
                c_high = c_set - c_low
                temp_final = c_high.union(c_verified).union(d_sig)

                if len(temp_final) > n_max:
                    # Raise bar for "High Attn" and retry
                    eta = next_stricter_spike(c_scores, eta)
                    continue
                else:
                    k_final = temp_final
                    break

            # Case B: Insufficient
            elif n_curr < n_min:
                n_needed = n_min - len(c_set)
                # Fill from rejected Disagreement tokens based on their Delta PPL
                # Sort d_remain by s_d score
                sorted_d_remain = sorted(list(d_remain), key=lambda x: s_d.get(x, -999), reverse=True)
                d_fill = set(sorted_d_remain[:n_needed])

                k_final = c_set.union(d_sig).union(d_fill)
                break

            # Case C: Ideal
            else:
                k_final = c_set.union(d_sig)
                break

        if not k_final:  # Fallback if loop breaks weirdly
            k_final = c_set.union(d_sig)

        return k_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_file", type=str, required=True, help="Teacher extracted keywords jsonl")
    parser.add_argument("--student_file", type=str, required=True, help="Student extracted keywords jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--eval_model_path", type=str, required=True, help="Path to model used for PPL calc")

    # Params for Algo 1
    parser.add_argument("--k_ratio", type=float, default=0.2, help="Initial Top-K selection ratio")
    parser.add_argument("--rho_min", type=float, default=0.10, help="Min density")
    parser.add_argument("--rho_max", type=float, default=0.30, help="Max density")

    args = parser.parse_args()

    confirmer = KeywordConfirmer(args.eval_model_path)

    # Read files
    # We assume lines correspond 1:1. If not, use ID mapping.
    with open(args.teacher_file, 'r') as f_t, open(args.student_file, 'r') as f_s, open(args.output_file, 'w') as f_out:
        t_lines = f_t.readlines()
        s_lines = f_s.readlines()

        assert len(t_lines) == len(s_lines), "Teacher and Student files must have same line count"

        for i, (line_t, line_s) in enumerate(tqdm(zip(t_lines, s_lines), total=len(t_lines), desc="Confirming")):
            data_t = json.loads(line_t)
            data_s = json.loads(line_s)

            # Copy base structure
            final_data = copy.deepcopy(data_s)

            segments = final_data['meta_info']['segments']

            # We process each segment that has 'key_attention_indices'
            # Note: T and S should have same segment structure

            for seg_idx, seg in enumerate(segments):
                if 'key_attention_indices' not in seg: continue

                # Get Teacher and Student Info
                # We expect data_t to have same structure
                seg_t = data_t['meta_info']['segments'][seg_idx]
                seg_s = data_s['meta_info']['segments'][seg_idx]

                # Construct Importance Vectors (Maps: Index -> Score)
                v_t = dict(zip(seg_t['key_attention_indices'], seg_t['key_attention_values']))
                v_s = dict(zip(seg_s['key_attention_indices'], seg_s['key_attention_values']))

                # Prepare Inputs for PPL
                # 1. Reconstruct Text for this segment (Plan)
                plan_text = seg['content']  # Or prefix + content
                plan_inputs = confirmer.tokenizer(plan_text, return_tensors='pt', add_special_tokens=False)
                plan_ids = plan_inputs.input_ids.to(confirmer.device)

                # 2. Find corresponding code (Target)
                # We need the code that follows this plan.
                # Heuristic: look at next segment. If type is 'code', use it.
                target_ids = torch.empty((1, 0), dtype=torch.long, device=confirmer.device)
                if seg_idx + 1 < len(segments):
                    next_seg = segments[seg_idx + 1]
                    if next_seg['type'] == 'code':
                        code_text = next_seg['content']
                        target_ids = confirmer.tokenizer(code_text, return_tensors='pt',
                                                         add_special_tokens=False).input_ids.to(confirmer.device)

                if target_ids.size(1) == 0:
                    # No code follows, cannot calc Delta PPL useful for code gen.
                    # Fallback: Union of T and S or skip
                    seg['final_pos_indices'] = list(set(v_t.keys()) | set(v_s.keys()))
                    continue

                L = plan_ids.size(1)

                # --- RUN ALGORITHM 1 ---
                final_pos_set = confirmer.dynamic_filter(
                    v_s, v_t,
                    None,
                    plan_ids, target_ids,
                    L,
                    args.k_ratio, args.rho_min, args.rho_max
                )

                # Define Negatives: (Union of original T/S candidates) - Positives
                all_candidates = set(v_t.keys()) | set(v_s.keys())
                final_neg_set = all_candidates - final_pos_set

                seg['key_pos_indices'] = list(final_pos_set)
                seg['key_neg_indices'] = list(final_neg_set)

                # Clean up original intermediate keys if needed
                # del seg['key_attention_indices']

            f_out.write(json.dumps(final_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()