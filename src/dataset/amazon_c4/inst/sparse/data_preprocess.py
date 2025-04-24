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


PROMPT = """Your task is to generate a structured Boolean search query based on the user's natural language user query.
Below is the user's natural language query:
```{user_query}```"""

def make_prefix(dp, template_type):
    input_str = PROMPT.format(user_query=dp['query'])
    if template_type == 'qwen-instruct':
        input_str = """<|im_start|>system\nYou are an AI assistant specializing in converting natural language user queries into structured Boolean search queries. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n""" + input_str
        input_str += """\nShow your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. The generated query should use Boolean operators (AND, OR) to structure your query logically. For example,
<answer>
{
    "query": xxx
}
</answer>.<|im_end|>
<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    else:
        raise NotImplementedError

    return input_str

def load_rec_dataset(data_dir):
    # load train.json
    with open(os.path.join(data_dir, 'train.json'), 'r') as f:
        train_data = json.load(f)
    
    # load test.json
    with open(os.path.join(data_dir, 'test.json'), 'r') as f:
        test_data = json.load(f)

    val_data_1 = []
    # select the last 10% of the training data as validation data
    val_data_1 = train_data[-int(0.05 * len(train_data)):]
    train_data = train_data[:-int(0.05 * len(train_data))]
    
    # select random 10% of the test data as validation data_2
    val_data_2 = random.sample(test_data, int(0.1 * len(test_data)))
    
    return train_data, val_data_1, val_data_2, test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/amazon_c4')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='qwen-instruct', choices=['base', 'qwen-instruct', 'gpt'])
    parser.add_argument('--save_dir', type=str, default='data/amazon_c4/inst')

    args = parser.parse_args()
    
    save_dir = os.path.join(args.save_dir, args.template_type)
    os.makedirs(save_dir, exist_ok=True)

    train_data, val_data_1, val_data_2, test_data = load_rec_dataset(args.local_dir)

    train_dataset = Dataset.from_list(train_data)
    val_1_dataset = Dataset.from_list(val_data_1)
    val_2_dataset = Dataset.from_list(val_data_2)
    test_dataset = Dataset.from_list(test_data)


    def make_map_fn(split, data_source):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['item_id'],
            }
            data = {
                "data_source": data_source,
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
    
    train_dataset = train_dataset.map(function=make_map_fn('train', 'amazon_c4_train'), with_indices=True)
    val_1_dataset = val_1_dataset.map(function=make_map_fn('val', 'amazon_c4_val'), with_indices=True)
    val_2_dataset = val_2_dataset.map(function=make_map_fn('val', 'amazon_c4_test_200'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test', 'amazon_c4_test'), with_indices=True)

    # shuffle the dataset
    train_dataset = train_dataset.shuffle(seed=42)

    # concatenate val_1 and val_2
    val_dataset = concatenate_datasets([val_1_dataset, val_2_dataset])
    
    threshold = 256
    
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
