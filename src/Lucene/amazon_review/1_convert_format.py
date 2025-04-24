import json
import os

def convert_jsonl_for_pyserini(input_file, output_file):
    """Convert JSONL data to Pyserini-compatible format with a structured 'contents' field"""
    docs = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())

            # # Construct a properly formatted "contents" field
            # contents = (
            #     f"Title: {data.get('title', '')}\n"
            #     f"Store: {data.get('store', '')}\n"
            #     f'Features: {" | ".join(data.get("features", []))}\n'
            #     f'Description: {" ".join(data.get("description", ""))}\n'
            #     f"Main Category: {data.get('main_category', '')}\n"
            #     f"Categories: {', '.join(data.get('categories', []))}\n"
            #     f"Details: {' | '.join(f'{k}: {v}' for k, v in data.get('details', {}).items())}\n"
            #     f"Average Rating: {data.get('average_rating', 'N/A')}\n"
            # )

            # Create JSON document with a clear structure
            doc = {
                "id": data["item_id"],  # Unique identifier for search results
                "contents": data['metadata'],  # Required field for Pyserini
                # "features": data.get("features", []),
                # "description": data.get("description", ""),
                # "title": data.get("title", ""),
                # "store": data.get("store", ""),
                # "main_category": data.get("main_category", ""),
                # "categories": data.get("categories", []),
                # "details": data.get("details", {}),
                # "average_rating": data.get("average_rating", 0.0),
            }

            docs.append(json.dumps(doc))

    with open(output_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc + "\n")

    print(f"✅ Converted JSONL saved to {output_file}")


domain_name_list = ['All_Beauty', 'Baby_Products', 'Video_Games']

for domain_name in domain_name_list:
    ori_data_dir = f"data/amazon_review/processed/corpus/{domain_name}_items_filtered.jsonl"
    output_file = f"database/amazon_review/{domain_name}/jsonl_docs/pyserini.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Example Usage
    convert_jsonl_for_pyserini(ori_data_dir, output_file)
