import argparse
import json
import os
import re
from tqdm import tqdm
import pdb

import sys
sys.path.append('./')

from src.eval_search.utils import ndcg_at_k
from src.Lucene.amazon_c4.search import PyseriniMultiFieldSearch


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', type=str, default='results/Baby/amazon_c4/eval_results_rec-r1.json')
    args = parser.parse_args()

    search_system = PyseriniMultiFieldSearch(index_dir='database/amazon_c4/pyserini_index')

    with open(args.res_path, 'r') as f:
        res_dict = json.load(f)

    
    test_data = []
    for _, value_dict in res_dict.items():
        query = str(value_dict['generated_text'])
        try:
            query = extract_answer(query)
        except:
            query = query
        query = str(query)
        target = value_dict['target']
        test_data.append({'query': query, 'target': target})
    
    ndcg = []
    batch_size = 100
    
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        queries = [item['query'] for item in batch]
        targets = {item['query']: item['target'] for item in batch} 
        
        results = search_system.batch_search(queries, top_k=100, threads=16)
        
        for query in queries:
            retrieved = [result[0] for result in results.get(query, [])]
            ndcg.append(ndcg_at_k(retrieved, targets[query], 10))
    
    print(f"Average NDCG@10: {sum(ndcg) / len(ndcg)}")