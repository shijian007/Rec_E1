import numpy as np
import re
import json

def dcg_at_k(retrieved, target, k, rel_scores=None):
    """
    Compute DCG@k (Discounted Cumulative Gain).
    Default target is an ordered list of relevant documents, from highest to lowest relevance.
    """
    retrieved = retrieved[:k]
    if rel_scores is None:
        gains = np.array(retrieved) == target
    else:
        assert len(target) == len(rel_scores)
        rel_scores_dict = {item: rel_scores[i] for i, item in enumerate(target)}
        gains = np.array([rel_scores_dict.get(doc, 0) for doc in retrieved])
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return np.sum(gains / discounts)

def ndcg_at_k(retrieved, target, k, rel_scores=None):
    """
    Compute NDCG@k.
    """
    dcg = dcg_at_k(retrieved, target, k, rel_scores)
    if isinstance(target, list):
        ideal_dcg = dcg_at_k(target, target, k, rel_scores)
    else:
        ideal_dcg = dcg_at_k([target], target, k, rel_scores)  # Ideal DCG: only the target at top
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(retrieved_list, target_list, k):
    if not target_list:
        return 0.0

    top_k = retrieved_list[:k]
    retrieved_relevant = set(top_k) & set(target_list)
    recall = len(retrieved_relevant) / len(set(target_list))
    return recall

def extract_answer(generated_text):
    # extract from \nAssistant:
    # try:
    #     generated_text = generated_text.split("\nAssistant:")[1]
    # except:
    #     generated_text = generated_text.split("\nassistant:")[1]
    
    # findall <answer> </answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, generated_text, re.DOTALL)  # Use re.DOTALL to match multiline content
    
    if len(matches) > 0:
        generated_text = matches[-1]
        try:
            # json.loads(generated_text)
            generated_text = json.loads(generated_text)['query']
        except:
            generated_text = matches[-1]

    return generated_text