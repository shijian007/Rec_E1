"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import re
import os
from datasets import Dataset, load_dataset, concatenate_datasets
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
from collections import defaultdict, Counter
import random
import pdb
import csv

PROMPT = """You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently. Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.
Below is the user purchase history:
```{purchase_history}```"""

def make_prefix(dp):
    input_str = PROMPT.format(purchase_history=dp['history_text'])
    input_str = """<|im_start|>system\nYou are a helpful AI assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\n""" + input_str
    input_str += """\nShow your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. The generated query should use Boolean operators (AND, OR) to structure your query logically. For example,
<answer>
{
    "query": xxx
}
</answer>.<|im_end|>
<|im_start|>assistant\nLet me solve this step by step.\n<think>"""

    return input_str

def load_rec_dataset(data_dir, domain_name_list, meta_dir='data/amazon_review/processed/corpus', history_trunct_num=10):
    train_data_dict = {}
    val_data_dict = {}
    test_data_dict = {}

    global_item2meta = {}

    for domain_name in domain_name_list:
        with open(os.path.join(data_dir, domain_name, 'train.json'), 'r') as f:
            train_data_dict[domain_name] = json.load(f)
        with open(os.path.join(data_dir, domain_name, 'val.json'), 'r') as f:
            val_data_dict[domain_name] = json.load(f)
        with open(os.path.join(data_dir, domain_name, 'test.json'), 'r') as f:
            test_data_dict[domain_name] = json.load(f)
        item_meta_path = os.path.join(meta_dir, f"{domain_name}_items_filtered.jsonl")
        with open(item_meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                global_item2meta[line['item_id']] = line['title']

    def add_history_text(data_list, item2meta):
        for entry in data_list:
            history = entry['history_list'][-history_trunct_num:] 
            entry['history_list'] = history
            history_text = [f'Item {idx + 1}: ' + item2meta.get(item_id, "") for idx, item_id in enumerate(history)]
            entry['history_text'] = "\n".join(history_text).strip()
        return data_list
    
    
    for domain_name in test_data_dict:
        train_data_dict[domain_name] = add_history_text(train_data_dict[domain_name], global_item2meta)
        val_data_dict[domain_name] = add_history_text(val_data_dict[domain_name], global_item2meta)
        test_data_dict[domain_name] = add_history_text(test_data_dict[domain_name], global_item2meta)

    return train_data_dict, val_data_dict, test_data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/amazon_review/ori_split_files')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--save_dir', type=str, default='data/amazon_review/inst')
    
    args = parser.parse_args()
    
    domain_name_list = ['Video_Games', 'Baby_Products', 'All_Beauty']
    
    data_source = f'amazon_review'
    data_source_dict = {ele: f'amazon_review_{ele}' for ele in domain_name_list}

    file_dir = args.local_dir
    save_dir = os.path.join(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    train_data_dict, val_data_dict, test_data_dict = load_rec_dataset(file_dir, domain_name_list)

    train_dataset_dict = {}
    val_dataset_dict = {}    
    test_dataset_dict = {}
    for domain_name in domain_name_list:
        train_data = train_data_dict[domain_name]
        train_dataset_dict[domain_name] = Dataset.from_list(train_data)

        val_data = val_data_dict[domain_name]
        val_dataset_dict[domain_name] = Dataset.from_list(val_data)

        test_data = test_data_dict[domain_name]
        test_dataset_dict[domain_name] = Dataset.from_list(test_data)
    
    
    def make_map_fn(split, data_source):
        def process_fn(example, idx):
            question = make_prefix(example)
            solution = {
                "target": example['target'],
            }
            data = {
                "data_source": data_source + f'_{split}',
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "amazon_review",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    

    for domain_name in domain_name_list:
        train_dataset = train_dataset_dict[domain_name]
        train_dataset_dict[domain_name] = train_dataset.map(function=make_map_fn('train', data_source_dict[domain_name]), with_indices=True)

        val_dataset = val_dataset_dict[domain_name]
        val_dataset_dict[domain_name] = val_dataset.map(function=make_map_fn('val', data_source_dict[domain_name]), with_indices=True)

        test_dataset = test_dataset_dict[domain_name]
        test_dataset_dict[domain_name] = test_dataset.map(function=make_map_fn('test', data_source_dict[domain_name]), with_indices=True)

    # concat val_dataset, all test_dataset
    test_dataset = concatenate_datasets(list(test_dataset_dict.values()))
    # # shuffle the dataset
    train_dataset = concatenate_datasets(list(train_dataset_dict.values()))
    val_dataset = concatenate_datasets(list(val_dataset_dict.values()))
    
    # concatenate val_1 and val_2
    val_dataset = concatenate_datasets([val_dataset, test_dataset])

    threshold = 512
    
    def truncate(train_dataset, threshold):
        count = 0
        # for those that are exceeding the threshold, we can delete the text between "\nTitle:" to "\nInclusion criteria:"
        for i, d in enumerate(train_dataset):
            if len(d['prompt'][0]['content'].split()) > threshold:
                text = d['prompt'][0]['content']
                count += 1
                words = text.split()
                truncate_length = max(threshold - 200, 0)  # Ensure we don't end up with a negative index
                text = ' '.join(words[-truncate_length:])
                train_dataset[i]['prompt'][0]['content'] = text

        print(f"Truncated {count} examples")
        
        return train_dataset
    
    train_dataset = truncate(train_dataset, threshold=threshold)

    hdfs_dir = os.path.join(args.hdfs_dir, args.template_type) if args.hdfs_dir is not None else None

    train_dataset.to_parquet(os.path.join(save_dir, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(save_dir, 'val.parquet'))
    test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'))
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=save_dir, dst=hdfs_dir) 
