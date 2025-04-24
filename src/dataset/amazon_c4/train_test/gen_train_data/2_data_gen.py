import json
import argparse
import os
import re
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
import time
import random
import glob

import sys
sys.path.append('./')
from src.utils.gpt_azure import gpt_chat_4omini
from src.dataset.amazon_c4.train_test.gen_train_data.utils import load_item_metadata_dict

# Define prompt
PROMPT = """
You are given a user review and the corresponding item metadata. Your task is to determine if the review is relevant to the item and, if so, rephrase the review into a first-person query that captures the essence of the user's needs and concerns.

Instructions:
1. Relevance Check: Analyze the review text and determine whether it is relevant to the item. Consider the review content only and avoid directly relying on item metadata unless the review provides no identifiable product context.
If the review clearly pertains to the product described in the metadata, output <relevance>1</relevance>; otherwise, output <relevance>0</relevance>.

2. Rephrased Query: If the review is relevant (<relevance>1</relevance>), transform the review into a natural, first-person query that maintains the review's intent and expresses the user's needs as if they were searching for the product.
The query should not contain explicit product names, brands, or technical specifications unless they are explicitly mentioned in the review and are unavoidable.
Keep the query detailed and natural, ensuring it reflects the user's thought process when searching for such a product.
Wrap the generated query in <query> </query> tags.

Input review:
```{review_text}```

Item metadata:
```{item_metadata}```
"""

def process_reviews(input):
    batch, item_metadata_dict, output_dir, process_id = input
    """Process a batch of reviews and save to a unique output file."""
    output_file = os.path.join(output_dir, f"filtered_reviews_{process_id}.jsonl")

    
    for idx, review in enumerate(tqdm(batch)):
        review_text = review['title'] + '\n' + review['text']
        item_metadata = item_metadata_dict.get(review['parent_asin'], "No metadata available")
        prompt = PROMPT.format(review_text=review_text, item_metadata=item_metadata)

        attempts = 0
        query = None

        while attempts < 3:
            try:
                attempts += 1
                response = gpt_chat_4omini(prompt)
                # Extract <relevance> and <query> tags
                relevance_match = re.findall(r'<relevance>(\d+)</relevance>', response)
                query_match = re.findall(r'<query>(.*?)</query>', response)

                if relevance_match and int(relevance_match[0]) == 1:
                    query = query_match[0]
                    break
            except:
                print(f"Error processing review {idx} in attempt {attempts}")
                attempts += 1

        # Save the query if found
        if query:
            output_dict = {
                "qid": idx,
                "query": query,
                "item_id": review['parent_asin'],
                "user_id": review['user_id'],
                "ori_rating": review['rating'],
                "ori_review": review['text']
            }
            with open(output_file, "a", encoding="utf-8") as file:
                file.write(json.dumps(output_dict) + "\n")
        
        


def chunk_list(data, num_chunks):
    """Split data into chunks for multiprocessing."""
    chunk_size = len(data) // num_chunks
    return [data[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', type=str, default='Baby')
    parser.add_argument('--data_dir', type=str, default='data/amazon_c4/raw/filtered_reviews')
    parser.add_argument('--output_dir', type=str, default='data/amazon_c4/gen_train/raw')
    parser.add_argument('--meta_data_path', type=str, default='data/amazon_c4/raw/sampled_item_metadata_1M.jsonl')
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()
    
    # Load metadata
    item_metadata_dict = load_item_metadata_dict(args.meta_data_path)
    
    save_dir = os.path.join(args.output_dir, args.domain_name)
    
    # Create output directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(args.data_dir, f"{args.domain_name}.jsonl")

    # Read filtered reviews
    with open(file_path, "r", encoding="utf-8") as file:
        filtered_reviews = [json.loads(line.strip()) for line in file]

    # random select 20000 reviews, seed=42
    random.seed(42)
    filtered_reviews = random.sample(filtered_reviews, 20000)

    # Split data into chunks for multiprocessing
    review_batches = chunk_list(filtered_reviews, args.num_workers)

    review_chuncks = [(batch, item_metadata_dict, save_dir, i) for i, batch in enumerate(review_batches)]

    # Create a multiprocessing pool
    with multiprocessing.Pool(args.num_workers) as pool:
        pool.map(process_reviews, review_chuncks)
