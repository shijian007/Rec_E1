import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pdb




# read data/esci/raw/sampled_item_metadata_esci.jsonl
with open('data/amazon_c4/raw/sampled_item_metadata_1M.jsonl', 'r') as file:
    sampled_metadata = [json.loads(line) for line in file]


doc_ids = []

for item in tqdm(sampled_metadata):
    doc_ids.append(item['item_id'])

# save to a numpy to data/esci/raw/esci/doc_ids.npy
np.save('data/amazon_c4/raw/cache/Amazon-C4', doc_ids)
