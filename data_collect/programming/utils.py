import os
import gzip
import json
import openai
import jsonlines
import re
from typing import List
from transformers import GPT2Tokenizer, AutoTokenizer


openai.api_key = os.getenv("OPENAI_API_KEY")
IMPORT_HEADER = "from typing import *\nimport math\nfrom heapq import *\nimport itertools\nimport re\nimport typing\nimport heapq\n_str=str\nimport re\nimport hashlib\nimport heapq\nimport collections\nfrom collections import *\nfrom itertools import combinations\nfrom math import prod\nfrom itertools import combinations_with_replacement\nfrom  decimal import Decimal, getcontext\nimport numpy as np\n"


def prepare_function_from_generated_code(dataset_type, prompt, generated_program, entry_point, add_header = True):
    if dataset_type in ["HumanEval", "MBPP", "APPS", "CodeContests","LiveCode"]:
        if (prompt in generated_program) or (('def ' + entry_point + '(') in generated_program):
            # It has the function header, no need to add
            cur_func_impl = generated_program
            print("yesyes")
        else:
            cur_func_impl = prompt + "\n" + generated_program
        # Add auxilary function
        cur_func_impl=filter_func(cur_func_impl)
        funcs = get_function(prompt)
        seed_funcs = [func[0] for func in get_function(generated_program)]
        #for func in funcs:
            #if func[0] not in seed_funcs:
                #cur_func_impl = func[1] + "\n" + cur_func_impl
                #print("nono")
        # Add comments

        if not find_comment(cur_func_impl, entry_point):
            cur_func_impl = fix_func_impl_comments(cur_func_impl, prompt, entry_point)
            print("nonono")
    # Add import
    # print(f"cur_func_impl111222331{cur_func_impl}1212321321")
    if add_header and IMPORT_HEADER not in cur_func_impl:
        cur_func_impl = IMPORT_HEADER + cur_func_impl
        # print(f"加脑袋cur_func_impl111222331{cur_func_impl}1212321321")
    assert isinstance(cur_func_impl, str)
    return cur_func_impl




def capture_import_statements(code: str) -> list:
    """
    Capture all import statements from the given Python code.

    Args:
    code (str): The string containing the Python code.

    Returns:
    list: A list of import statements found in the code.
    """
    matches = re.findall(r'^\s*(import\s+\w+(\s*,\s*\w+)*|from\s+\w+(\.\w+)*\s+import\s+\w+(\s*,\s*\w+)*)', code,
                         re.MULTILINE)
    return [match[0] for match in matches]

def extract_docstring(function_str: str) -> str:
    # Regular expression to find the content between triple quotes
    docstring_pattern = re.compile(r'""".*?"""', re.DOTALL)

    # Search for the pattern in the function string
    match = docstring_pattern.search(function_str)

    if match:
        # Extract and clean up the docstring
        docstring = match.group(0)
        docstring = docstring.strip('"""')
        return docstring.strip()
    else:
        return ""

def contains_test_case(line: str) -> bool:
    # Define the regular expression pattern to search for "test case"
    test_case_pattern = re.compile(r'test case', re.IGNORECASE)

    # Check if the line is a print statement
    if line.strip().startswith('print('):
        return False

    # Search for the pattern in the line
    return bool(re.search(test_case_pattern, line))

def contains_assert(line: str) -> bool:
    # Define the regular expression pattern to search for "test case"
    assert_pattern = re.compile(r'assert', re.IGNORECASE)

    # Check if the line is a print statement
    if line.strip().startswith('print('):
        return False

    # Search for the pattern in the line
    return bool(re.search(assert_pattern, line))

def contains_fix(line: str) -> bool:
    # Define the regular expression pattern to search for "test case"
    start_fix_explanation=re.compile(r'Fixing Explanation',re.IGNORECASE)

    # Check if the line is a print statement
    if line.strip().startswith('print('):
        return False

    # Search for the pattern in the line
    return bool(re.search(start_fix_explanation, line))

def contains_adjust(line: str) -> bool:
    # Define the regular expression pattern to search for "test case"
    start_explanation_adjustments = re.compile(r'Explanation Adjustments', re.IGNORECASE)

    # Check if the line is a print statement
    if line.strip().startswith('print('):
        return False

    # Search for the pattern in the line
    return bool(re.search(start_explanation_adjustments, line))


import re


def filter_func(func_imp):
    """
    过滤和清洗生成的代码，保留函数体、必要的注释（Plan）和 Imports。
    修复了因 Plan 注释中包含 'test case' 导致被错误截断的问题。
    """
    # 1. 提取文档字符串和 import（保持原有逻辑）
    problem_description = extract_docstring(func_imp)
    import_statements = capture_import_statements(func_imp)

    # 去除 Markdown 标记
    func_imp = func_imp.replace("```python", "").replace("```", "")

    # 【重要】不要使用 func_imp.find("def") 切片，否则 def 前面的 Plan 会被删掉
    # func_imp = func_imp[func_imp.find("def"):]

    func_lines = []

    # 预编译正则（保持原有逻辑）
    start_program_pattern = re.compile(r'Start Program', re.IGNORECASE)
    end_program_pattern = re.compile(r'End Program', re.IGNORECASE)
    start_fixed_program_pattern = re.compile(r'Start Fixed Program', re.IGNORECASE)
    end_fixed_program_pattern = re.compile(r'End Fixed Program', re.IGNORECASE)

    lines = func_imp.split("\n")

    # 1. 找到 def 所在的行号
    def_line_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            def_line_index = i
            break

    if def_line_index == -1:
        def_line_index = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # --- 处理 def 之前的内容 ---
        if i < def_line_index:
            # 只保留注释(#)和装饰器(@)，跳过可能的废话文本
            if stripped.startswith("#") or stripped.startswith("@") or stripped == "":
                func_lines.append(line)
            continue

        # --- 处理 def 及之后的内容 ---

        # 跳过特定的标记行
        if (re.search(start_program_pattern, line) or
                re.search(end_program_pattern, line) or
                re.search(start_fixed_program_pattern, line) or
                re.search(end_fixed_program_pattern, line)):
            continue

        # 【核心修复】
        # 遇到测试用例或断言时停止，但【忽略注释行】。
        # 您的 Plan 注释包含 "test case" 字样，如果这里不检查 is_comment，就会被误判为测试用例从而 break。
        is_comment = stripped.startswith("#")

        if not is_comment and line not in problem_description and (
                contains_test_case(line) or contains_assert(line) or contains_fix(line) or contains_adjust(line)):
            # 只有当这一行 不是注释 且 包含停止关键词 时，才中断
            break

        func_lines.append(line)

    # 重新组合：Imports + 过滤后的函数体
    func_lines = import_statements + func_lines

    func_imp = "\n".join(func_lines)
    return func_imp


def solution_plan_process(list_of_plan):
    modified_plan = []
    start_plan=r'\"*\[*(?i)Start Plan\]*\"*'
    end_plan=r'\"*\[*(?i)End Plan\]*\"*'
    for plan in list_of_plan:
        if re.search(start_plan,plan) and re.search(end_plan,plan):
            start_index = plan.find("[Start Plan]") + len("[Start Plan]")
            end_index = plan.find("[End Plan]")
            solution_plan = plan[start_index:end_index]
            modified_plan.append(solution_plan)
        else:
            tmp_plan = ""
            for line in plan.split("\n"):
                if line == "":
                    continue
                if line[0].isdigit():
                    tmp_plan += line + "\n"
            modified_plan.append(tmp_plan)
    return modified_plan

def solution_plan_filter(solution_plan):
    start_plan = r'\"*\[*(Start Plan)\]*\"*'
    end_plan   = r'\"*\[*(End Plan)\]*\"*'

    if re.search(start_plan, solution_plan, re.IGNORECASE) and \
       re.search(end_plan, solution_plan, re.IGNORECASE):

        matches_start = list(re.finditer(start_plan, solution_plan, re.IGNORECASE))
        matches_end   = list(re.finditer(end_plan, solution_plan, re.IGNORECASE))

        start_index = matches_start[0].end()
        end_index   = matches_end[0].start()

        solution_plan = solution_plan[start_index:end_index]

    return solution_plan


import re

def evaluation_message_filter(evaluation_message, tests_i):
    evaluation_regex   = r'\"*\[*(Verification for)\]*\"*'
    evaluation_regex_2 = r"Let's verify"
    evaluation_regex_3 = r'\"*\[*(Correct Plan)\]*\"*'

    evaluation_list = evaluation_message.split("\n")
    evaluation_save_list = []

    count = 0
    for each_line in evaluation_list:
        # 1. 跳过 verification 起始说明
        if re.search(evaluation_regex_2, each_line, re.IGNORECASE) or \
           re.search(evaluation_regex_3, each_line, re.IGNORECASE):
            continue

        # 2. 匹配 Verification for
        if re.search(evaluation_regex, each_line, re.IGNORECASE):
            if count < len(tests_i):
                evaluation_save_list.append(f"[Verification for {tests_i[count]}]")
                count += 1
            else:
                # 防止 tests_i 越界（很重要）
                evaluation_save_list.append("[Verification for UNKNOWN]")
        else:
            evaluation_save_list.append(each_line)

    return "\n".join(evaluation_save_list)



def program_analysis_filter(analysis_words):
    my_analysis_regex=r'(?i)\"*\[*My Analysis\]*\"*'
    matches_start = re.finditer(my_analysis_regex, analysis_words)
    match_indices_start = [(match.start(), match.end()) for match in matches_start]

    start_index=0
    if len(match_indices_start)!=0:
        start_index = match_indices_start[0][1]


    error_analysis=analysis_words[start_index:]
    return error_analysis

def explain_filter(explain):
    start_explain_regex=r'\"*\[*(?i)Start Explanation\]*\"*'
    end_explain_regex=r'\"*\[*(?i)End Explanation\]*\"*'
    matches_start = re.finditer(start_explain_regex, explain)
    match_indices_start = [(match.start(), match.end()) for match in matches_start]

    start_index=0
    if len(match_indices_start)!=0:
        start_index = match_indices_start[0][1]


    matches_end = re.finditer(end_explain_regex, explain)
    match_indices_end = [(match.start(), match.end()) for match in matches_end]

    end_index=len(explain)-1
    if len (match_indices_end)!=0:
        end_index = match_indices_end[0][0]

    program_explain=explain[start_index:end_index]

    return program_explain



def print_information_filter(tokenizer, print_information):
    if len(print_information)>25000:
        total_tokenizer=0
        part_length = 2500
        part_information_ahead=[]
        print_information_list=print_information.split("\n")
        for line in  print_information_list:
            total_tokenizer+=len(tokenizer.tokenize(line))
            if total_tokenizer<=part_length:
                part_information_ahead.append(line)
            else:
                break

        part_information_ahead.append("...")
        part_information_tail=[]
        total_tokenizer=0
        for line in  reversed(print_information_list):
            total_tokenizer+=len(tokenizer.tokenize(line))
            if total_tokenizer<=part_length:
                part_information_tail.append(line)
            else:
                break
        print_information = "\n".join(part_information_ahead + part_information_tail[::-1])
        return print_information

    else:

        if len(tokenizer.tokenize(print_information))<=5000:
            return print_information
        else:
            total_tokenizer=0
            part_length = 2500
            part_information_ahead=[]
            print_information_list=print_information.split("\n")
            for line in  print_information_list:
                total_tokenizer+=len(tokenizer.tokenize(line))
                if total_tokenizer<=part_length:
                    part_information_ahead.append(line)
                else:
                    break

            part_information_ahead.append("...")
            part_information_tail=[]
            total_tokenizer=0
            for line in  reversed(print_information_list):
                total_tokenizer+=len(tokenizer.tokenize(line))
                if total_tokenizer<=part_length:
                    part_information_tail.append(line)
                else:
                    break

            print_information="\n".join(part_information_ahead+part_information_tail[::-1])

        return print_information

import re

import re


def reformat_imp_tags(text):
    """
    Reformats text by moving <imp:x> tags from the end of a block to immediately
    after the specific section headers ([STEP:x], [GEN_PLAN], etc.).

    It also removes the trailing newline/whitespace from the content that was
    previously separating the content from the <imp> tag.
    """

    # 定义我们需要查找的起始标签
    # [STEP:\d+] 匹配 [STEP:1], [STEP:2] 等
    # [GEN_GLOBAL_PLAN], [GEN_PLAN], [GEN_CODE] 匹配对应的关键字
    target_tags = r"(?:GEN_GLOBAL_PLAN|STEP:\d+|GEN_PLAN|GEN_CODE)"

    # 编译正则表达式
    # 逻辑说明:
    # 1. (\[...\]): 捕获组1 - 匹配起始标签 (如 [STEP:1])
    # 2. (.*?): 捕获组2 - 非贪婪匹配中间的所有内容（包括换行符）
    # 3. (<imp:\d+>): 捕获组3 - 匹配结尾的 importance tag (如 <IMP:5>), 忽略大小写
    pattern = re.compile(
        rf'(\[{target_tags}\])(.*?)(<imp:\d+>)',
        re.DOTALL | re.IGNORECASE
    )

    def replacer(match):
        tag = match.group(1)  # 例如: [STEP:1]
        content = match.group(2)  # 例如: \nParse the input...\n
        imp_tag = match.group(3)  # 例如: <IMP:5>

        # 【核心修改】
        # 使用 rstrip() 去除内容末尾的空白字符（包括换行符）。
        # 这就是“顺便删掉那一个空行”的操作。
        # 原始结构通常是: Content + \n + <IMP>
        # 移走 <IMP> 后，如果不处理，就会多出一个悬空的 \n。
        clean_content = content.rstrip()

        # 拼接新格式：标签 + 空格 + IMP标记 + 清理后的内容
        return f"{tag} {imp_tag}{clean_content}"

    # 执行替换
    result = pattern.sub(replacer, text)
    return result

class DataPostProcessor:
    def __init__(self):
        # 匹配所有 GEN 标签，包括 _REVISED 后缀
        # 捕获组 () 会在 split 后保留分隔符
        self.tag_pattern = re.compile(r'(\[GEN_[A-Z_]+\])')

        # 匹配 risk 标记，用于判断该块是否包含错误和反思
        self.risk_pattern = re.compile(r'\[risk\]', re.IGNORECASE)

    def process(self, full_response: str):
        """
        输入: 模型生成的完整 Diff 文本
        输出: (raw_data, clean_data)
           - raw_data: 原始完整文本 (用于训练反思能力)
           - clean_data: 纯净代码/计划 (用于代码验证)
        """

        # 1. 第一个变量：保存原始信息
        raw_data = full_response.strip()

        # 2. 第二个变量：清洗数据
        clean_data = self._clean_response(full_response)

        return raw_data, clean_data

    def _clean_response(self, text: str) -> str:
        # 使用正则分割文本
        # split 后列表格式通常为: [前导文, TAG_1, Content_1, TAG_2, Content_2, ...]
        parts = self.tag_pattern.split(text)

        clean_segments = []

        # 处理前导文（如果有的话，通常是 Problem Description，视情况保留）
        # 这里假设我们需要保留 Tag 之前的内容（如题目描述），如果不需要可以跳过 parts[0]
        if parts[0].strip():
            clean_segments.append(parts[0])

        # 从索引 1 开始遍历，每次步进 2 (Tag, Content)
        for i in range(1, len(parts), 2):
            tag = parts[i].strip()
            # 确保不越界
            content = parts[i + 1] if i + 1 < len(parts) else ""

            # 判断逻辑：

            # 情况 A: 修正后的块 ([GEN_..._REVISED])
            if "_REVISED]" in tag:
                # 1. 还原标签名：去掉 _REVISED (例如 [GEN_CODE_REVISED] -> [GEN_CODE])
                standard_tag = tag.replace("_REVISED]", "]")
                # 2. 保留内容 (修正后的内容里通常没有 risk 标签)
                clean_segments.append(f"{standard_tag}{content}")

            # 情况 B: 包含错误的块 (含有 [risk])
            elif self.risk_pattern.search(content):
                # 1. 这是一个错误块，它的 content 里包含了错误代码 + [risk] + [solution]
                # 2. 我们直接丢弃整个块，不仅删除了错误代码，也删除了反思信息
                continue

            # 情况 C: 正确的原始块 (既不是 REVISED 也没有 risk)
            else:
                # 保留原始内容
                clean_segments.append(f"{tag}{content}")

        return "".join(clean_segments).strip()

def revised_solution_plan(solution_plan):
    regex_pattern_start = r'(?i)\[*Start Revised Solution Plan]\]*'
    regex_pattern_end = r'(?i)\[*End Revised Solution Plan]\]*'
    if  not re.search(regex_pattern_start, solution_plan):
        assert "incorrect Revised Plan"

    matches_start = re.finditer(regex_pattern_start, solution_plan)
    match_indices_start = [(match.start(), match.end()) for match in matches_start]
    start_plan_index=0
    if len(match_indices_start)!=0:
        start_plan_index = match_indices_start[0][1]


    matches_end = re.finditer(regex_pattern_end, solution_plan)
    match_indices_end = [(match.start(), match.end()) for match in matches_end]
    end_plan_index=len(solution_plan)-1
    if len (match_indices_end)!=0:
        end_plan_index = match_indices_end[0][0]

    revised_plan = solution_plan[start_plan_index:end_plan_index]
    return revised_plan
def parse_analysis_data(analysis_text):
    """
    解析错误分析文本，返回一个错误列表。
    每个元素包含: {'type': 'plan'/'code', 'index': 1, 'risk': '...', 'solution': '...'}
    """
    errors = []
    # 匹配 <Local Plan X> 或 <Code Segment X>
    # 及其后的 [risk]... [solution]...
    pattern = re.compile(
        r'<(Local Plan|Code Segment)\s+(\d+)>\s*'
        r'\[risk\]:\s*(.*?)\s*'
        r'\[solution\]:\s*(.*?)(?=(?:<Local Plan|<Code Segment|Execution Logic|$))',
        re.DOTALL | re.IGNORECASE
    )

    for match in pattern.finditer(analysis_text):
        tag_type = match.group(1)  # "Local Plan" or "Code Segment"
        index = int(match.group(2))
        risk = match.group(3).strip()
        solution = match.group(4).strip()

        # 归一化类型标识，方便后续匹配
        block_type = 'GEN_PLAN' if 'Plan' in tag_type else 'GEN_CODE'

        errors.append({
            'target_type': block_type,
            'target_index': index,  # 这里的 index 是第几个 Plan 或 第几个 Code
            'risk': risk,
            'solution': solution
        })
    return errors

# reflection
def parse_plan_code_blocks(raw_text):
    """
    将原始的 Plan-Code 文本拆解为块列表。
    """
    # 定义主要标签
    tags = ['[GEN_GLOBAL_PLAN]', '[GEN_PLAN]', '[GEN_CODE]', '[EOS]']
    # 构建正则分割模式
    split_pattern = f"({'|'.join(map(re.escape, tags))})"

    parts = re.split(split_pattern, raw_text)

    blocks = []
    current_tag = None

    # 维护计数器
    plan_counter = 0
    code_counter = 0

    # 遍历分割结果 (parts[0]通常为空或前导文本, parts[1]是标签, parts[2]是内容...)
    for part in parts:
        if not part.strip():
            continue

        if part in tags:
            current_tag = part
            # 只有遇到具体的 Plan 或 Code 标签才增加计数
            if current_tag == '[GEN_PLAN]':
                plan_counter += 1
                curr_idx = plan_counter
            elif current_tag == '[GEN_CODE]':
                code_counter += 1
                curr_idx = code_counter
            else:
                curr_idx = 0  # Global Plan 或 EOS 不参与计数匹配

            blocks.append({
                'tag': current_tag,
                'content': '',  # 内容稍后填充
                'type': current_tag.strip('[]'),  # 去掉括号作为类型
                'index': curr_idx
            })
        else:
            # 这是内容部分，追加到最近一个 block
            if blocks:
                blocks[-1]['content'] += part

    return blocks


def generate_reflection_samples(plan_code_text, analysis_text):
    """
    主函数：生成反思数据列表
    """
    # 1. 解析数据
    plan_code_blocks = parse_plan_code_blocks(plan_code_text)
    errors = parse_analysis_data(analysis_text)

    reflection_samples = []

    # 2. 循环处理每一个错误
    for error in errors:
        current_sample_text = ""
        found_target = False

        # 3. 遍历 Block 构建当前样本
        for block in plan_code_blocks:
            # 拼接当前块 (Tag + Content)
            current_sample_text += f"{block['tag']}{block['content']}"

            # 4. 检查是否到达了错误发生的位置
            # 必须类型相同 (GEN_PLAN vs GEN_PLAN) 且 序号相同 (第2个 vs 第2个)
            if block['tag'] == f"[{error['target_type']}]" and block['index'] == error['target_index']:
                # 5. 插入反思数据 (Risk & Solution)
                reflection_block = (
                    f"[RISK] [IMP:10]"
                    "{error['risk']}\n"
                    f"[SOLUTION]"
                    " {error['solution']}\n"
                )
                current_sample_text += reflection_block

                found_target = True
                break  # 关键：截断后续内容，跳出内部循环

        if found_target:
            reflection_samples.append(current_sample_text)
        else:
            print(f"Warning: Could not find matching block for {error['target_type']} #{error['target_index']}")

    return reflection_samples
def process_evaluation_results (message):

    evaluation_list=[]
    # regex_pattern = r'(?i)\[\"*?Verification for [\w\d\s\'()]+\"*\]'
    regex_pattern=r'(?i)\[+(Plan )?Verification for .*?\]+'
    compare_pattern=r'(?i)Results Compare'
    occurrences_verification = [m.start() for m in re.finditer(regex_pattern, message)]
    for i in range(len(occurrences_verification)):
        if i + 1 < len(occurrences_verification):
            evaluation_left_index = occurrences_verification[i]
            evaluation_right_index = occurrences_verification[i + 1]
            evaluation_information=message[evaluation_left_index:evaluation_right_index]
            evaluation_list_split=evaluation_information.split('\n')
            evaluation_temp_list=[]
            for each_line in evaluation_list_split:
                if re.search(regex_pattern,each_line):
                    continue
                if re.search(compare_pattern, each_line):
                    break
                evaluation_temp_list.append(each_line)
            part_evaluation_information="\n".join(evaluation_temp_list)
            evaluation_list.append(part_evaluation_information)

        else:
            evaluation_left_index=occurrences_verification[i]
            evaluation_information=message[evaluation_left_index:]

            evaluation_list_split=evaluation_information.split('\n')
            evaluation_temp_list=[]
            for each_line in evaluation_list_split:
                if re.search(regex_pattern,each_line):
                    continue
                if re.search(compare_pattern, each_line):
                    break
                evaluation_temp_list.append(each_line)
            part_evaluation_information="\n".join(evaluation_temp_list)
            evaluation_list.append(part_evaluation_information)

    return evaluation_list


def fix_func_impl_comments(func_impl: str, prompt: str, entry) -> str:
    """
    如果生成的代码缺少文档字符串，尝试从 prompt 中提取注释并插入。
    支持提取格式：
    1. 三引号文档字符串 ("""""" / '''...''')
    2. 以 # 开头的单行注释块
    """
    comments = None

    # 1. 尝试提取 Docstring (最高优先级)
    if prompt.find('\"\"\"') != -1:
        parts = prompt.split('\"\"\"')
        if len(parts) > 1:
            comments = parts[1]
    elif prompt.find('\'\'\'') != -1:
        parts = prompt.split('\'\'\'')
        if len(parts) > 1:
            comments = parts[1]

    # 2. 如果没找到 Docstring，尝试提取 # 开头的注释
    if comments is None:
        hash_comment_lines = []
        for line in prompt.split('\n'):
            stripped_line = line.strip()
            if stripped_line.startswith('#'):
                # 去掉开头的 # 和随后的空格
                content = stripped_line.lstrip('#').strip()
                hash_comment_lines.append(content)

        if hash_comment_lines:
            comments = '\n'.join(hash_comment_lines)

    # 如果完全没找到任何注释内容，直接返回原代码，避免插入空字符串
    if comments is None:
        return func_impl

    # 3. 寻找函数定义的插入点
    func_impl_lines = func_impl.split('\n')
    insert_index = -1

    for i, line in enumerate(func_impl_lines):
        # 使用 strip() 增加鲁棒性，防止 def 前面有缩进导致匹配失败
        if line.strip().startswith('def') and entry in line:
            insert_index = i
            break

    # 4. 插入注释
    if insert_index != -1:
        # 将提取到的内容包装成 Docstring 格式插入到 def 下一行
        # 注意：这里硬编码了4个空格缩进。如果代码风格不同可能需要调整。
        func_impl_lines.insert(insert_index + 1, '    \"\"\"' + comments + '\n    \"\"\"')
        return '\n'.join(func_impl_lines)
    else:
        # 如果找不到 def，返回原代码（或者您可以选择追加在文件头）
        return func_impl

def insert_comment(func_impl: str, comment: str, entry: str) -> str:
    func_impl_lines = func_impl.split('\n')
    for i, line in enumerate(func_impl_lines):
        if line.startswith('def ' + entry + '('):
            break
    func_impl_lines.insert(i + 1, '    \"\"\"' + comment + '\"\"\"')
    return '\n'.join(func_impl_lines)


import re


def find_comment(func_impl: str, entry: str) -> bool:
    """
    检测函数是否有注释（支持 Docstring 和 # 开头的注释）。
    兼容类型注解 (-> int) 和行尾注释。
    """
    # 1. 构建更强大的正则
    # def        : 匹配 def
    # \s+        : 匹配空格
    # entry      : 函数名
    # \s* : 可选空格
    # \(.*?\)    : 匹配参数部分 (非贪婪)
    # [^:]* : 【关键修改】匹配冒号前的任意字符（为了兼容 -> int 类型注解）
    # :          : 函数定义的结束冒号
    pattern = r"def\s+" + re.escape(entry) + r"\s*\(.*?\)[^:]*:"

    # 使用 DOTALL 模式，防止参数换行导致匹配失败
    match = re.search(pattern, func_impl, re.DOTALL)

    if not match:
        # 如果连函数定义都没找到，直接返回 False
        return False

    # 2. 获取函数定义“冒号”之后的所有内容
    body_start_index = match.end()
    func_body = func_impl[body_start_index:]

    # 3. 去掉前导空白（空格、换行、Tab）
    # 这一步非常关键：
    # 如果是行内注释：":# comment" -> lstrip() -> "# comment"
    # 如果是换行注释：":\n    # comment" -> lstrip() -> "# comment"
    trimmed_body = func_body.lstrip()

    # 4. 检测是否以注释符号开头
    if (trimmed_body.startswith('"""') or
            trimmed_body.startswith("'''") or
            trimmed_body.startswith("#")):
        return True

    return True


def get_function(prompt):
    lines = prompt.split('\n')
    funcs = []

    # 缓冲区：用于积累当前正在处理的行
    # 初始化时，它会收集第一个 def 之前的所有内容（如 imports, 注释）
    buffer_lines = []

    # 当前正在记录的函数名
    current_func_name = None

    for line in lines:
        # 检测是否是函数定义行
        # 使用 startswith 保持和你原代码一致，也可以改为 re.match(r"^\s*def ") 以支持缩进
        if line.startswith("def "):
            new_func_name = line.split("def ")[1].split("(")[0]

            if current_func_name:
                # === 遇到新函数，且之前已经在记录旧函数 ===
                # 这说明上一个函数结束了，新函数开始了。
                # 难点：buffer_lines 的末尾可能包含属于新函数的注释（如 # [GEN_PLAN]）
                # 我们需要从后往前“回溯”，把这些注释切分给新函数

                split_idx = len(buffer_lines)
                # 倒序遍历缓冲区，找到代码和注释的分界线
                for i in range(len(buffer_lines) - 1, -1, -1):
                    l_strip = buffer_lines[i].strip()
                    # 如果是注释(#)、装饰器(@)或者是空行，通常属于下一个函数的头部
                    if l_strip.startswith('#') or l_strip.startswith('@') or l_strip == "":
                        split_idx = i
                    else:
                        # 一旦遇到非注释的代码行（如 return x），停止回溯
                        break

                # 执行切分
                prev_func_body = buffer_lines[:split_idx]  # 归属上一个函数
                next_func_prefix = buffer_lines[split_idx:]  # 归属新函数（作为前缀）

                # 保存上一个函数
                if prev_func_body:
                    funcs.append([current_func_name, "\n".join(prev_func_body)])

                # 重置缓冲区：新函数前缀 + 当前 def 行
                buffer_lines = next_func_prefix + [line]
                current_func_name = new_func_name

            else:
                # === 遇到第一个函数 ===
                # 此时 buffer_lines 里是 import 或第一个 Plan
                # 直接将 def 行追加进去，它们整体属于第一个函数
                buffer_lines.append(line)
                current_func_name = new_func_name

        else:
            # === 非 def 行 ===
            # 无论是代码还是注释，先暂时放入缓冲区
            buffer_lines.append(line)

    # 循环结束，保存最后一个函数
    if current_func_name and buffer_lines:
        funcs.append([current_func_name, "\n".join(buffer_lines)])

    return funcs




def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    elif not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items += [item]
    return items

def read_jsonl_map(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    elif not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items = {}
    with jsonlines.open(path) as reader:
        for item in reader:
            items[item['task_id']] = item
    return items

def write_jsonl(path: str, data: List[dict], append: bool = False):
    with jsonlines.open(path, mode='a' if append else 'w') as writer:
        for item in data:
            writer.write(item)


def read_jsonl_gz(path: str) -> List[dict]:
    if not path.endswith(".jsonl.gz"):
        raise ValueError(f"File `{path}` is not a jsonl.gz file.")
    with gzip.open(path, "rt") as f:
        data = [json.loads(line) for line in f]
    return data



def replace_test(item, items_test):
    if item['task_id'] in items_test:
        item['given_tests'] = items_test[item['task_id']]['given_tests']
    else:
        item['given_tests'] = []
    return item

def enumerate_resume(dataset, results_path, testfile=None):
    items_test = {}

    if testfile is not None:
        print("testfile", testfile)
        items_test = read_jsonl_map(testfile)

    exist_items = []
    if os.path.exists(results_path):
        print(results_path)
        with jsonlines.open(results_path) as reader:
            for item in reader:
                exist_items.append(item['task_id'])

    for i, item in enumerate(dataset):

        # if item['task_id'] in exist_items:
        #     continue
        item = replace_test(item, items_test)
        yield i, item

def replace_seed_test(item, items_seed, items_test):
    if item['task_id'] in items_seed:
        item['seed'] = items_seed[item['task_id']]['solution']
        if 'is_passing' in items_seed[item['task_id']]:
            item['is_passing'] = items_seed[item['task_id']]['is_passing']
        else:
            item['is_passing'] = False
    else:
        item['seed'] = ""
    if item['task_id'] in items_test:
        item['given_tests'] = items_test[item['task_id']]['given_tests']
    else:
        item['given_tests'] = []
    return item



def count_solved(logpath) -> float:
    solved = 0
    count = 0
    dataset = open(logpath, "r")
    for l in dataset:
        item = json.loads(l)
        count += 1
        if "is_solved" in item and item["is_solved"]:
            solved += 1
    return float(solved) / count




