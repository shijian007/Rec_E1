import json
import argparse
import os
import re
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
import time
import random
import csv
import pdb

import sys
sys.path.append('./')


def parse_csv_qrel(csv_file_path):

    query_dict = {}

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) 

        for row in reader:
            qid, query, item_id = row

            if query not in query_dict:
                query_dict[query] = {"targets": []}

            query_dict[query]["targets"].append(item_id)

    return query_dict


def load_item_dict(meta_data_path):
    item_dict = {}
    with open(meta_data_path, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line.strip())
            item_dict[item["item_id"]] = item['category']
    
    return item_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_data_path', type=str, default='data/esci/raw/test.csv')
    parser.add_argument('--output_dir', type=str, default='data/esci/test_subset')
    parser.add_argument('--meta_data_path', type=str, default='data/esci/raw/sampled_item_metadata_esci.jsonl')
    args = parser.parse_args()
    
    target_subdomain = ['Video_Games', 'Baby_Products', 'Office_Products', 'Sports_and_Outdoors', 'Others']

    # Load metadata
    item_metadata_dict = load_item_dict(args.meta_data_path)

    # Load full data
    full_data = parse_csv_qrel(args.full_data_path)
    # make it a list
    full_data = [{"query": k, "item_id": v["targets"]} for k, v in full_data.items()]
    # save full data to test_subset/
    # with open(os.path.join(args.output_dir, "full_data.json"), "w") as f:
    #     json.dump(full_data, f, indent=4)
    
    sub_domain_data = defaultdict(list)

    for entry in tqdm(full_data):
        item_id = entry['item_id']
        item_category = item_metadata_dict[item_id[0]]
        if item_category in target_subdomain:
            sub_domain_data[item_category].append(entry)
        else:
            sub_domain_data["Others"].append(entry)

    os.makedirs(args.output_dir, exist_ok=True)
    for sub_domain, data in sub_domain_data.items():
        output_file = os.path.join(args.output_dir, f"{sub_domain}.json")
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

        # print the number of entries in each sub-domain
        print(f"Sub-domain: {sub_domain}, Number of entries: {len(data)}")

    full_train_data = parse_csv_qrel('data/esci/raw/train.csv')
    full_train_data = [{"query": k, "item_id": v["targets"]} for k, v in full_train_data.items()]
    with open(os.path.join(args.output_dir, "train.json"), "w") as f:
        json.dump(full_train_data, f, indent=4)
    
    print(f"Number of entries in train data: {len(full_train_data)}")