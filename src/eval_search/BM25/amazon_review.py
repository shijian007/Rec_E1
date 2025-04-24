import argparse
import json
import os
import re
from tqdm import tqdm
import numpy as np
from collections import defaultdict

import sys
sys.path.append('./')

from src.eval_search.utils import ndcg_at_k, recall_at_k
from src.Lucene.amazon_c4.search import PyseriniMultiFieldSearch


def extract_answer(generated_text):
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, generated_text, re.DOTALL)
    if matches:
        extracted = matches[-1]
        try:
            return json.loads(extracted)['query']
        except:
            return extracted
    return generated_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', type=str, default='database/amazon_c4/pyserini_index')
    parser.add_argument('--res_path', type=str, default='results/Baby/amazon_c4/eval_results_rec-r1.json')
    parser.add_argument('--save_path', type=str, default='results/Baby/amazon_c4/query_metric_results.json')
    args = parser.parse_args()

    search_system = PyseriniMultiFieldSearch(index_dir=args.index_dir)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    with open(args.res_path, 'r') as f:
        res_dict = json.load(f)

    test_data = []
    for sample_id, value_dict in res_dict.items():
        query = str(value_dict['generated_text'])
        try:
            query = extract_answer(query)
        except:
            query = query
        query = str(query)
        target = value_dict['target']
        test_data.append({'id': sample_id, 'query': query, 'target': target})

    topks = [10, 50]
    batch_size = 100

    results_dict = {}

    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        queries = [item['query'] for item in batch]
        ids = [item['id'] for item in batch]
        targets = {item['query']: item['target'] for item in batch}

        search_results = search_system.batch_search(queries, top_k=100, threads=16)

        for idx, query in enumerate(queries):
            sample_id = ids[idx]
            retrieved = [result[0] for result in search_results.get(query, [])]
            key = f"{sample_id}_{query}"

            results_dict[key] = {
                'target': targets[query],
                'retrieved': str(retrieved),
                'ndcg@10': ndcg_at_k(retrieved, targets[query], 10),
                'ndcg@50': ndcg_at_k(retrieved, targets[query], 50),
                'recall@10': recall_at_k(retrieved, [targets[query]], 10),
                'recall@50': recall_at_k(retrieved, [targets[query]], 50),
            }

    # Save results
    with open(args.save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Also print mean scores
    ndcg_10 = [v['ndcg@10'] for v in results_dict.values()]
    ndcg_50 = [v['ndcg@50'] for v in results_dict.values()]
    recall_10 = [v['recall@10'] for v in results_dict.values()]
    recall_50 = [v['recall@50'] for v in results_dict.values()]

    print(f"Recall@10: {np.mean(recall_10):.4f}")
    print(f"Recall@50: {np.mean(recall_50):.4f}")
    print(f"NDCG@10: {np.mean(ndcg_10):.4f}")
    print(f"NDCG@50: {np.mean(ndcg_50):.4f}")
