import re
import random, traceback
import numpy as np
import ast
import operator
import pdb
import json
from collections import defaultdict
import sys

sys.path.append('./')

from src.Lucene.mindrec.search_docs import PyseriniMultiFieldSearch

index_dir_dict = {
    'mindsmall_rec': 'Your pyserini_index path',
}
try:
    search_system_dict = {
        'mindsmall_rec': PyseriniMultiFieldSearch(index_dir=index_dir_dict['mindsmall_rec']),
    }
except Exception as e:
    print(e)
    print('[Error] Please build the index first:')


def dcg_at_k(retrieved, target, k):
    """
    Compute DCG@k (Discounted Cumulative Gain).
    """
    retrieved = retrieved[:k]
    gains = [1.0 if item == target else 0.0 for item in retrieved]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return np.sum(gains / discounts)


def ndcg_at_k(retrieved, target, k):
    """
    Compute NDCG@k.
    """
    dcg = dcg_at_k(retrieved, target, k)
    ideal_dcg = dcg_at_k([target], target, k)  # Ideal DCG: only the target at top
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(retrieved_list, target_list, k):
    if not target_list:
        return 0.0

    top_k = retrieved_list[:k]
    retrieved_relevant = set(top_k) & set(target_list)
    recall = len(retrieved_relevant) / len(set(target_list))
    return recall

# rec-r1
'''def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1].strip()
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1].strip()
    else:
        print("[Error] Failed to locate model response header")
        return None, processed_str
    # processed_str = solution_str

    # Regular expression to find the last occurrence of <answer>...</answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, processed_str, re.DOTALL)  # Use re.DOTALL to match multiline content

    if matches:
        return matches[-1].strip(), processed_str  # Return the last matched answer
    else:
        print("[Error] No valid answer tags found")
        return None, processed_str'''

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Regular expression to find the last occurrence of <answer>...</answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)  # Use re.DOTALL to match multiline content

    if matches:
        return matches[-1].strip(), solution_str  # Return the last matched answer
    else:
        print("[Error] No valid answer tags found")
        return None, solution_str


def validate_response_structure(processed_str: str, do_print: bool) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        if do_print:
            print(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            if do_print:
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
            positions['think_end'] > positions['answer_start'] or
            positions['answer_start'] > positions['answer_end']):
        if do_print:
            print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        if do_print:
            print("  Tag sequence validation passed")

    return validation_passed


def check_json_format(json_str, do_print=False):
    """Check if the given string is a valid JSON and follows the expected structure."""
    try:
        if not json_str:
            if do_print:
                print("[Error] Empty JSON string")
            return False

        data = json.loads(json_str)

        # Required keys
        required_keys = {"query"}
        if not all(key in data for key in required_keys):
            if do_print:
                print("[Error] Missing required keys in JSON")
            return False

        return True
    except json.JSONDecodeError:
        if do_print:
            print("[Error] JSON decoding failed")
        return False


def retriver_items(query, domain_name, top_k=3000, threads=16):
    """Retrieve items from the search system."""
    results = search_system_dict[domain_name].batch_search([query], top_k=top_k, threads=threads)
    return results


def calculate_answer_score(json_str, label, topk, domain_name):
    """Calculate answer score based on final_prediction idx."""
    try:
        data = json.loads(json_str)
        query = data['query']
        print("[Query] ", query)
        target = label
        # print("[Target] ", target)
        results = retriver_items(query, domain_name, top_k=topk, threads=32)
        asin_results = [item[0] for item in results[query]]
        # print("[asin_results] ", asin_results)
        answer_score = ndcg_at_k(asin_results, target, topk)
        print("[NDCG] ", answer_score)

    except Exception as e:
        print("ERR: %s\t%s" % (str(e), traceback.format_exc().replace('\n', '')))
        print("[Error] Error in evaluation")
        answer_score = -2

    return answer_score


def compute_score(solution_str, ground_truth, data_source, format_reward=1):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """

    label = str(ground_truth['target'])
    answer_text, processed_str = extract_solution(solution_str)
    do_print = random.randint(1, 32) == 1

    # Validate response structure
    response_format_correct = validate_response_structure(processed_str, do_print)
    json_format_correct = check_json_format(answer_text, do_print)
    format_correct = response_format_correct and json_format_correct
    format_score = format_reward if format_correct else -2

    if do_print:
        print(f"--------------------------------")
        print(f"[Solution string]: {solution_str}")
        print(f"[Target]: {label}")

    if 'mindsmall' in data_source:
        domain_name = 'mindsmall_rec'

    if 'test' in data_source or 'dev' in data_source or 'val' in data_source:
        top_k = 50
    else:
        top_k = 5000

    answer_score = 0
    if format_correct and answer_text:
        answer_score = calculate_answer_score(answer_text, label, top_k, domain_name) * 10

    if answer_score > 0:
        total_score = format_score + answer_score
    else:
        if format_score > 0:
            total_score = 0
        else:
            total_score = format_score

    if do_print:
        print("\n" + "-" * 80)
        print(f" Final Score ".center(80, '-'))
        print(f"  Format: {format_score}")
        print(f"  Answer: {answer_score}")
        print(f"  Total: {total_score}")
        print("=" * 80 + "\n")

    return total_score


if __name__ == '__main__':
    data_source = 'mindsmall_rec'
    json_string = '{"query": "证监会 AND IPO战略配售政策"}'

    solution_str = """<|im_start|>assistant: Here is the answer to your question: <think> </think> <answer>{"query": "(NOT \\"3-Pack Replacement for Whirlpool\\") AND Amazon home"}</answer>
"""
    ground_truth = {'target': 'NW202503283359614085'}

    score = compute_score(solution_str, ground_truth, data_source)
