import json
import argparse
import pandas as pd
from tqdm import tqdm
import pdb
import sys
sys.path.append('./')

from src.dataset.amazon_c4.train_test.gen_train_data.utils import load_item_metadata_dict


def filter_reviews(file_path):
    # Initialize an empty list to store filtered reviews
    filtered_reviews = []

    # Read and process the JSONL file
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Parse each line as a JSON object
            review = json.loads(line.strip())

            # Apply filtering conditions
            if (
                review["rating"] == 5.0  # Check if the rating is 5.0
                and review["verified_purchase"] is True  # Check if purchase is verified
                and len(review["text"]) >= 100  # Check if the review text has at least 100 characters
            ):
                filtered_reviews.append(review)

    return filtered_reviews


def filter_out_existing(csv_file_path, filtered_reviews, item_metadata_dict):
    df = pd.read_csv(csv_file_path)
    existing_pairs = set(zip(df["item_id"], df["user_id"]))
    initial_count = len(filtered_reviews)
    filtered_reviews = [
        review for review in tqdm(filtered_reviews) if ((review["parent_asin"], review["user_id"]) not in existing_pairs) and (review["parent_asin"] in item_metadata_dict)
    ]
    removed_count = initial_count - len(filtered_reviews)
    print(f"Removed {removed_count} existing reviews from the dataset.")
    return filtered_reviews


file_path_dict = {
    'Baby': 'data/amazon_review/raw/Baby/Baby_Products.jsonl',
    'Video_Games': 'data/amazon_review/raw/Video_Games/Video_Games.jsonl',
    'Office': 'data/amazon_review/raw/Office/Office_Products.jsonl',
    'Sports': 'data/amazon_review/raw/Sports/Sports_and_Outdoors.jsonl'
}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file_path', type=str, default='data/amazon_c4/raw/test.csv')
    parser.add_argument('--meta_data_path', type=str, default='data/amazon_c4/raw/sampled_item_metadata_1M.jsonl')
    args = parser.parse_args()

    item_metadata_dict = load_item_metadata_dict(args.meta_data_path)
    
    
    for category, file_path in file_path_dict.items():
        review_list = []
        print(f"Filtering reviews for {category} category")
        # Filter reviews
        filtered_reviews = filter_reviews(file_path)
        # Filter out existing reviews
        filtered_reviews = filter_out_existing(args.csv_file_path, filtered_reviews, item_metadata_dict)

        review_list.extend(filtered_reviews)
        
        # Save the filtered reviews to a JSONL file
        with open(f"data/amazon_c4/raw/filtered_reviews/{category}.jsonl", "w", encoding="utf-8") as file:
            for review in review_list:
                file.write(json.dumps(review) + "\n")
    
        # print counts of reviews
        print(f"Total reviews: {len(review_list)}")