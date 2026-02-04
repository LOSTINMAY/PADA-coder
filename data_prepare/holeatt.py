import json
import torch
from transformers import AutoConfig, AutoTokenizer
from colorama import Fore, Back, Style, init

# 初始化颜色
init(autoreset=True)


def inspect_full_context_highlight(jsonl_file, model_path):
    print(f"{Fore.CYAN}Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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


    print(f"{Fore.CYAN}Reading first line from {jsonl_file}...")
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        line = f.readline()
        if not line:
            return
        data = json.loads(line)

    print(f"{Fore.YELLOW}ID: {data.get('id', 'Unknown')}")

    # --- 1. 重构完整的 Token 序列 (必须与提取脚本一致) ---
    # 提取 System + User
    # 注意：这里假设 messages[0]=System, messages[1]=User。根据你的构造逻辑调整。
    if len(data['messages']) >= 2:
        msg_0 = data['messages'][0]['content']
        msg_1 = data['messages'][1]['content']
        base_prompt = f"{msg_0}\n{msg_1}\n"
    else:
        base_prompt = data['messages'][0]['content'] + "\n"

    full_text_accumulated = base_prompt

    # 计算初始 Offset
    base_ids = tokenizer(full_text_accumulated, add_special_tokens=True).input_ids
    current_offset = len(base_ids)

    # 保存完整的 input_ids 列表
    all_input_ids = list(base_ids)

    segments = data['meta_info']['segments']
    segment_ranges = []

    # 循环构建完整序列
    for i, seg in enumerate(segments):
        seg_text = f"{seg['prefix']}\n{seg['content']}"
        full_text_with_seg = full_text_accumulated + seg_text + "\n"

        new_ids = tokenizer(full_text_with_seg, add_special_tokens=True).input_ids

        start = current_offset
        end = len(new_ids)

        segment_ranges.append({
            "type": seg['type'],
            "start": start,
            "end": end,
            "data": seg
        })

        full_text_accumulated = full_text_with_seg
        all_input_ids = new_ids  # 更新
        current_offset = end

    # --- 2. 可视化输出 ---
    print(f"\n{Fore.GREEN}{'=' * 20} ATTENTION HEATMAP RECONSTRUCTION {'=' * 20}")

    # 我们只展示那些包含注意力数据的 Segment (通常是 Code 或 Local Plan)
    for idx, seg_info in enumerate(segment_ranges):
        seg_data = seg_info['data']

        # 如果这个段落没有提取过注意力，就跳过 (或者它是不可训练的)
        if 'key_attention_indices' not in seg_data:
            continue

        print(f"\n{Fore.MAGENTA}{'=' * 10} View from Segment {idx}: {seg_info['type'].upper()} {'=' * 10}")
        print(
            f"{Fore.LIGHTBLACK_EX}Explanation: White text is context. {Fore.RED}Red text{Fore.LIGHTBLACK_EX} is where this segment looked.")
        print("-" * 60)

        # 获取注意力数据
        indices_set = set(seg_data['key_attention_indices'])
        # 做一个映射：index -> score
        index_to_score = {}
        if 'key_attention_values' in seg_data:
            for i, tok_idx in enumerate(seg_data['key_attention_indices']):
                if i < len(seg_data['key_attention_values']):
                    index_to_score[tok_idx] = seg_data['key_attention_values'][i]

        # --- 核心：逐个 Token 打印历史 ---
        # 我们的关注范围是：从 0 到 当前段落的开始 (seg_info['start'])
        # 因为代码是在看它“之前”的东西

        context_end = seg_info['start']

        # 为了输出流畅，我们积攒字符串
        output_buffer = ""

        for i in range(context_end):
            token_id = all_input_ids[i]
            token_str = tokenizer.decode([token_id])

            # 处理换行，让输出好看点
            display_str = token_str.replace('\n', ' ')

            if i in indices_set:
                # 【高亮逻辑】
                score = index_to_score.get(i, 0.0)
                # 红色加粗 + 原词 + 分数下标
                # \033[4m 是下划线，Style.BRIGHT 是加粗
                highlighted = f"{Style.BRIGHT}{Fore.RED}{token_str}{Style.RESET_ALL}{Fore.CYAN}₍{score:.1f}₎{Style.RESET_ALL}"
                print(highlighted, end="")  # 直接打印，不换行
            else:
                # 普通文本：灰色，降低存在感，突出红色
                print(f"{Fore.LIGHTWHITE_EX}{token_str}{Style.RESET_ALL}", end="")

        print("\n" + "-" * 60)
        print(f"{Fore.GREEN}>>> GENERATING CURRENT SEGMENT ({seg_info['type']}):")

        # 打印当前正在生成的段落（用绿色表示）
        current_tokens = all_input_ids[seg_info['start']:seg_info['end']]
        current_text = tokenizer.decode(current_tokens)
        print(f"{Fore.GREEN}{current_text}")
        print("\n")


if __name__ == "__main__":
    # 替换你的路径
    """
    JSONL_FILE = "/root/autodl-tmp/attentioncode/data/Qwen3-8B/test.jsonl"  # 你的输出文件
    MODEL_PATH = "/root/autodl-tmp/attentioncode/qwen2.5-coder-7B"
    """
    JSONL_FILE = "E:/study/yanyi/论文/自己的小论文/attentioncode/data/Qwen3-8B/test.jsonl"  # 你的输出文件
    MODEL_PATH = "E:/study/yanyi/论文/自己的小论文/attentioncode/Qwen3-8B"

    inspect_full_context_highlight(JSONL_FILE, MODEL_PATH)