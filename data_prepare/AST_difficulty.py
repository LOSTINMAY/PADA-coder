import argparse
import json
import ast
import os


class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.node_count = 0
        self.max_depth = 0
        self.control_flow_count = 0
        self.current_depth = 0

    def generic_visit(self, node):
        self.node_count += 1
        self.current_depth += 1
        if self.current_depth > self.max_depth:
            self.max_depth = self.current_depth

        # Check whether it is a control flow statement
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try,
                             ast.ExceptHandler, ast.With, ast.FunctionDef)):
            self.control_flow_count += 1

        super().generic_visit(node)
        self.current_depth -= 1


def heuristic_complexity(code_str):
    """
    An alternative estimation scheme when AST parsing fails
    """
    keywords = ['if', 'for', 'while', 'def', 'class', 'try', 'except', 'with']
    count = sum(code_str.count(kw) for kw in keywords)
    max_indent = 0
    for line in code_str.splitlines():
        if not line.strip(): continue
        indent = len(line) - len(line.lstrip())
        max_indent = max(max_indent, indent)

    score = (count * 0.1) + (max_indent / 16.0)
    return round(min(score, 1.0), 4)


def calculate_complexity_score(code_str):
    """
    Calculate the normalized complexity score (0.0 - 1.0)
    """
    if not code_str or not code_str.strip():
        return 0.0

    try:
        try:
            tree = ast.parse(code_str)
        except IndentationError:
            # Fallback: wrap in a dummy function to handle loose code blocks
            padded_code = "def dummy():\n" + "\n".join(["    " + line for line in code_str.splitlines()])
            tree = ast.parse(padded_code)
        except SyntaxError:
            return heuristic_complexity(code_str)

        visitor = ComplexityVisitor()
        visitor.visit(tree)

        score_nodes = min(visitor.node_count / 100.0, 1.0)
        score_depth = min(visitor.max_depth / 5.0, 1.0)
        score_control = min(visitor.control_flow_count / 10.0, 1.0)

        # Weighted score
        raw_score = 0.2 * score_nodes + 0.4 * score_depth + 0.4 * score_control
        return round(min(raw_score, 1.0), 4)

    except Exception:
        return heuristic_complexity(code_str)


def main():
    parser = argparse.ArgumentParser(description="Calculate Code Complexity Scores")
    parser.add_argument("--input_file", type=str, required=True, help="Input .jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Output .jsonl file")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    print(f"Processing: {args.input_file} -> {args.output_file}")
    processed_count = 0

    with open(args.input_file, 'r', encoding='utf-8') as f_in, \
            open(args.output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            if not line.strip():
                continue

            data = json.loads(line)

            if 'meta_info' in data and 'segments' in data['meta_info']:
                for seg in data['meta_info']['segments']:
                    if seg.get('type') == 'code':
                        complexity = calculate_complexity_score(seg['content'])
                        seg['complexity'] = complexity

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            processed_count += 1

    print(f"Total processed lines: {processed_count}")


if __name__ == "__main__":
    main()