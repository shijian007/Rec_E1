import json
import argparse
import os
import re
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
import time
import random
import pdb

import sys
sys.path.append('./')


def load_item_dict(meta_data_path):
    item_dict = {}
    with open(meta_data_path, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line.strip())
            item_dict[item["item_id"]] = item['category']
    
    return item_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_data_path', type=str, default='data/amazon_c4/test.json')
    parser.add_argument('--output_dir', type=str, default='data/amazon_c4/test_subset')
    parser.add_argument('--meta_data_path', type=str, default='data/amazon_c4/raw/sampled_item_metadata_1M.jsonl')
    args = parser.parse_args()
    
    target_subdomain = ['Games', 'Baby', 'Office', 'Sports']

    # Load metadata
    item_metadata_dict = load_item_dict(args.meta_data_path)

    # Load full data
    with open(args.full_data_path, "r") as f:
        full_data = json.load(f)
    
    other_domain_data = []

    for entry in tqdm(full_data):
        item_id = entry['item_id']
        item_category = item_metadata_dict[item_id]
        if item_category not in target_subdomain:
            try:
                other_domain_data.append(entry)
            except:
                pdb.set_trace()
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"others.json")
    with open(output_file, "w") as f:
        json.dump(other_domain_data, f, indent=4)

    # print the number of entries in each sub-domain
    print(f"Sub-domain: others, Number of entries: {len(other_domain_data)}")