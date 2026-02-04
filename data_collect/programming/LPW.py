from generators import PyGenerator, model_factory
from executors import PyExecutor
from filelock import FileLock
from collections import defaultdict
from utils import *
from transformers import GPT2Tokenizer
import json
import os
import time
import re

from typing import *
from bisect import bisect_left, bisect_right

def extract_fixed_program(text: str) -> str:
    pattern = r"\[Start Fixed Program\](.*?)\[End Fixed Program\]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def uncomment_keep_indent(text):
    """
# code ->     code
    """
    lines = text.split('\n')
    processed_lines = []

    for line in lines:
        new_line = re.sub(r'(^\s*)#\s?', r'\1', line)
        processed_lines.append(new_line)

    return "\n".join(processed_lines)

def convert_codegen_to_python(raw_text):
    """
    Convert text containing tags such as [GEN_PLAN], [GEN_CODE], etc. into Python code.
rules
    1. The content below [GEN_CODE] is reserved as code.
    2. All other tags ([GEN_PLAN], [STEP], [EOS]) and their contents are converted to comments.
    """
    lines = raw_text.split('\n')
    processed_lines = []

    is_code_block = False

    for line in lines:
        stripped_line = line.strip()

        if '[GEN_CODE]' in stripped_line:
            is_code_block = True
            processed_lines.append(f"# {line}")
            continue


        if any(tag in stripped_line for tag in ['[GEN_PLAN]', '[GEN_GLOBAL_PLAN]', '[STEP:', '[EOS]']):
            is_code_block = False
            processed_lines.append(f"# {line}")
            continue

        if is_code_block:
            processed_lines.append(line)
        else:
            if stripped_line:
                processed_lines.append(f"# {line}")
            else:
                processed_lines.append(line)

    return "\n".join(processed_lines)

def save_intermediate_plan(item, plan_content, plan_type, iteration):
    """
    Write the generated plan into a file in real-time
    :param item: Dictionary of question information
    :param plan_content: The text content of the plan
    :param plan_type: 'initial' (initial) or 'revised' (revised)
    :param iteration: current iteration roun
    """
    record = {
        "task_id": item["task_id"],
        "prompt": item["prompt"],
        "entry_point": item["entry_point"],
        "plan": plan_content,
        "type": plan_type,
        "iter": iteration,
        "timestamp": time.time()
    }

    with FileLock("intermediate_plans.jsonl.lock"):
        with open("intermediate_plans.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # print(f"  [Saved] {plan_type} plan for {item['task_id']} iter {iteration}")

def parse_analysis(response_text):
    """
Analyze the model output, extract [Algorithm], [Strategy], and [Complexity], and return them separately.
Usage example:
    algorithm, strategy, complexity = analyzer.parse_analysis(model_output)
    """
    algorithm = ""
    strategy = ""
    complexity = ""

    algo_match = re.search(r"\[Algorithm\]:\s*(.+)", response_text, re.IGNORECASE)
    if algo_match:
        algorithm = algo_match.group(1).strip()

    strat_match = re.search(r"\[Strategy\]:\s*(.+?)(?=\n\[Complexity\]|$)", response_text,
                                re.IGNORECASE | re.DOTALL)
    if strat_match:
        strategy = strat_match.group(1).strip()

    comp_match = re.search(r"\[Complexity\]:\s*(.+)", response_text, re.IGNORECASE)
    if comp_match:
        complexity = comp_match.group(1).strip()

    return algorithm, strategy, complexity

def plan_generation(item, model_name, max_iters, port):
    model = model_factory(model_name, port)
    gen = PyGenerator()
    cur_Exp = 0
    dataset_type = item["task_id"].split("/")[0]
    tests_i = item["given_tests"]
    tests_i = [test for test in tests_i if item['entry_point'] in test and 'assert False' not in test]
    tests_words = ""
    for test in tests_i:
        tests_words += test + "\n"

    if model.is_chat:
        messages = []
    else:
        messages = ""

    token_num = 0

    ALGO, message = gen.ALGO_generation(item["prompt"], item["canonical_solution"], item["entry_point"], model,
                                                  messages, dataset_type)

    algorithm, strategy, complexity = parse_analysis(ALGO)
    print(f"[{item['task_id']}] Teacher ({model_name}) generating plan...")
    generated_plan, message = gen.plan_generation(item["prompt"], item["entry_point"], algorithm, strategy, complexity, item["canonical_solution"], model,
                                                  messages, dataset_type)
    generated_plan = reformat_imp_tags(generated_plan)

    plan_solution = solution_plan_filter(generated_plan)

    save_intermediate_plan(item, plan_solution, "initial", 0)
    # ==========================

    plan_results_list = set()

    collected_plan_data = {
        "initial_plan": plan_solution,
        "refinements": []
    }

    while cur_Exp < max_iters:
        print(f"  - Plan verification iter {cur_Exp}...")
        plan_verification, message = gen.plan_evaluation(item["prompt"], item["entry_point"], plan_solution,
                                                         tests_words, model, messages, dataset_type)


        # Record the verification information for each round
        current_refinement = {
            "iter": cur_Exp,
            "plan": plan_solution,
            "verification": plan_verification
        }
        # if there is no tag [Revised Solution Plan] in evaluation message, the evaluation success.
        if plan_verification.count("Verification for") == len(tests_i) and plan_verification.count(
                "Revised Solution Plan") == 0:

            # check [Verification for] tag
            plan_evaluation_for_each_test = evaluation_message_filter(plan_verification, tests_i)

            # re-check the evaluation message
            verification_check_message, message = gen.plan_evaluation_check(item["prompt"], item["entry_point"],
                                                                            plan_solution,
                                                                            plan_evaluation_for_each_test,
                                                                            model, messages, dataset_type)


            # pass the self-check
            if verification_check_message.count("Correct Analysis") == len(tests_i):
                verification_list = process_evaluation_results(plan_verification)
                if len(verification_list) != len(tests_i):
                    cur_Exp += 1
                    continue

                # success
                plan_results_list = (plan_solution, plan_verification, verification_list)
                collected_plan_data["final_plan"] = plan_solution
                collected_plan_data["final_verification"] = plan_verification
                collected_plan_data["success"] = True
                break

        elif (plan_verification.count("Verification for") > len(tests_i) and plan_verification.count(
                "Revised Solution Plan") == 0) or (
                plan_verification.count("Verification for") < len(tests_i) and plan_verification.count(
                "Revised Solution Plan") == 0):
            cur_Exp += 1
            continue
        else:
            plan_solution = revised_solution_plan(plan_verification)
            current_refinement["revised_plan"] = plan_solution

            save_intermediate_plan(item, plan_solution, "revised", cur_Exp + 1)
            # ==============================

        collected_plan_data["refinements"].append(current_refinement)
        cur_Exp += 1

    if not plan_results_list:
        collected_plan_data["success"] = False

    return plan_results_list, tests_i, token_num, 0, 0, 0, collected_plan_data


def code_generation(i, item, log_path, model_name, max_iters, plan_results_list, test_cases, token_num,
                    collected_plan_data, port):
    exe = PyExecutor()
    model = model_factory(model_name, port)

    dataset_type = item["task_id"].split("/")[0]

    # Code data collect
    collected_code_data = {
        "attempts": [],
        "final_solution": None,
        "success": False
    }

    if len(plan_results_list) == 0:
        return

    gen = PyGenerator()
    if model.is_chat:
        messages = []
    else:
        messages = ""

    incorrect_program_record = []
    code_generation_plan = plan_results_list[0]
    code_generation_verification_list = plan_results_list[2]

    # 构建 Prompt
    code_generation_evaluation_prompt = "\n".join(
        [f"[Plan Verification for {test_cases[i]}]\n\n{code_generation_verification_list[i]}" for i in
         range(len(code_generation_verification_list))])

    print(f"[{item['task_id']}] Teacher ({model_name}) generating code...")
    generated_program, message = gen.code_generation(item["prompt"], item["entry_point"],
                                                     code_generation_evaluation_prompt, code_generation_plan,
                                                     model, messages, dataset_type)
    generated_program = reformat_imp_tags(generated_program)

    cleaned_program = convert_codegen_to_python(generated_program)



    cur_func_impl_without_print = prepare_function_from_generated_code(dataset_type, item["prompt"], cleaned_program,
                                                                       item["entry_point"], add_header=False)

    # Add print
    generated_program_with_print, _ = gen.print_generation(cur_func_impl_without_print,
                                                           code_generation_evaluation_prompt, model, messages,
                                                           dataset_type)


    cur_func_impl_with_print = prepare_function_from_generated_code(dataset_type, item["prompt"],
                                                                    generated_program_with_print, item["entry_point"])
    # print(f"cur_func_impl_with_print\n{cur_func_impl_with_print}\nend")

    cur_Exp = 0
    while cur_Exp < max_iters:
        print(f"  - Code execution/debug iter {cur_Exp}...")
        # evaluate on sample tests
        is_passing, failed_tests, printed_output, reward, timeout_example_test, failed_tests_list, failed_printed_output_list = exe.execute(
            cur_func_impl_with_print, test_cases)

        current_attempt = {
            "iter": cur_Exp,
            "code": generated_program,
            "is_passing": is_passing,
            "failed_tests": failed_tests_list
        }

        if is_passing:
            print(f"  - Success! Found golden solution.")
            collected_code_data["final_solution"] = generated_program
            collected_code_data["success"] = True
            collected_code_data["attempts"].append(current_attempt)
            break
        else:
            # Error Analysis

            failed_tests_case = test_cases[failed_tests_list[0]]
            correct_verification = plan_results_list[2][failed_tests_list[0]]
            failed_printed_output = print_information_filter(model.tokenizer, failed_printed_output_list[0])
            cur_func_impl_without_print = uncomment_keep_indent(cur_func_impl_without_print)

            # 1. Code Explain
            #generated_program_explain, _ = gen.program_explain(item["prompt"], cur_func_impl_without_print, model,messages, dataset_type)
            #program_explain = explain_filter(generated_program_explain)

            # 2. Error Analysis
            # print(f"cur_func_impl_with_print{cur_func_impl_with_print}cur_func_impl_with_printcur_func_impl_with_print")
            cur_func_impl_with_print = uncomment_keep_indent(cur_func_impl_with_print)
            program_analysis, _ = gen.program_analysis(item["prompt"], cur_func_impl_without_print, item["canonical_solution"],
                                                       failed_tests_case, failed_printed_output, model, messages,
                                                       dataset_type)
            error_analysis = program_analysis_filter(program_analysis)

            current_attempt["error_analysis"] = error_analysis
            # current_attempt["program_explain"] = program_explain

            incorrect_history_prompt = "[Incorrect History]\n\n"
            for kvs in incorrect_program_record:
                incorrect_history_prompt += f"[History Error Program]\n\n{kvs}\n\n"

            # 3. Refine Code
            correct_func_impl_without_print, _ = gen.correct_program(item["prompt"], cur_func_impl_without_print,
                                                                      error_analysis,
                                                                     incorrect_history_prompt, model, messages,
                                                                     dataset_type)
            correct_func_impl_without_print = extract_fixed_program(correct_func_impl_without_print)
            correct_func_impl_without_print = convert_codegen_to_python(correct_func_impl_without_print)

            cur_func_impl_without_print_imp = prepare_function_from_generated_code(dataset_type, item["prompt"],
                                                                               correct_func_impl_without_print,
                                                                               item["entry_point"])

            correct_func_impl_with_print, _ = gen.print_generation(cur_func_impl_without_print_imp,
                                                                   code_generation_evaluation_prompt, model, messages,
                                                                   dataset_type)
            cur_func_impl_with_print = prepare_function_from_generated_code(dataset_type, item["prompt"],
                                                                            correct_func_impl_with_print,
                                                                            item["entry_point"])

        collected_code_data["attempts"].append(current_attempt)
        cur_Exp += 1

    # === save data ===
    if collected_code_data["success"]:
        golden_entry = {
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "entry_point": item["entry_point"],
            "plan_data": collected_plan_data,
            "code_data": collected_code_data,
        }

        with FileLock("golden_data_APPStest.jsonl.lock"):
            with open("golden_data_APPStest.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(golden_entry, ensure_ascii=False) + "\n")
        print(f"[{item['task_id']}] Golden data saved.")


def programming_task(i, item, log_path, model_name, max_iters, port):
    plan_results_list, test_cases, _, _, _, _, collected_plan_data = plan_generation(item, model_name, max_iters, port)

    if plan_results_list:
        code_generation(i, item, log_path, model_name, max_iters, plan_results_list, test_cases, 0, collected_plan_data,
                        port)


def run_lpw(
        dataset: List[dict],
        model_name: str,
        max_iters: int,
        log_path: str,
        verbose: bool,
        testfile: str = None,
        port: str = "",
) -> None:
    num_items = len(dataset)
    args = iter([(i, item, log_path, model_name, max_iters, port) for i, item in
                 enumerate_resume(dataset, log_path,testfile)])
    for item in args:
        print(f'==start {item[0]+1}/{num_items}')
        programming_task(*item)
    print("Accuracy:", count_solved(log_path))
