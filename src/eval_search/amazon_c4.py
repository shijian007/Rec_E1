import argparse
import json
import re
import os
import pandas as pd
import pdb
import sys
sys.path.append('./')
from src.eval_search.utils import ndcg_at_k
from src.Lucene.amazon_c4.search import PyseriniMultiFieldSearch
from tqdm import tqdm

def extract_query_target(data):
    query_target_pairs = []
    
    for key, value in data.items():
        # pdb.set_trace()
        processed_str = value['generated_text'].split("\nassistant", 1)[1].strip()
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(answer_pattern, processed_str, re.DOTALL)
        if matches:
            answer_json = matches[-1].strip()
            try:
                answer_data = json.loads(answer_json) 
                query = answer_data.get("query")
                target = value["target"]
                if query:
                    query_target_pairs.append((query, target))
            except json.JSONDecodeError:
                print(f"Error parsing JSON in key {key}")
    
    return query_target_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_file', type=str, default='results/amazon_c4/eval_results_rec_r1.json')
    parser.add_argument('--out_file', type=str, default='results/amazon_c4/eval_results_qwen2.5-3B-instruct.txt')
    args = parser.parse_args()

    with open(args.res_file, 'r') as f:
        data = json.load(f)

    query_target_pairs = extract_query_target(data)
    
    search_system = PyseriniMultiFieldSearch(index_dir="database/amazon_c4/pyserini_index")
    
    
    ndcg = []
    batch_size = 100

    for i in tqdm(range(0, len(query_target_pairs), batch_size)):
        batch = query_target_pairs[i:i+batch_size]
        queries = [item[0] for item in batch]
        targets = {item[0]: item[1] for item in batch} 
        
        results = search_system.batch_search(queries, top_k=100, threads=16)
        
        for query in queries:
            retrieved = [result[0] for result in results.get(query, [])]
            ndcg.append(ndcg_at_k(retrieved, targets[query], 100))

    print(f"Average NDCG@3000: {sum(ndcg) / len(ndcg)}")