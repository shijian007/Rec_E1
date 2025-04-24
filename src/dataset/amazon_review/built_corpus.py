import json
import os
import pdb

file_dir = 'data/amazon_review/processed'
domain_name_list = ['All_Beauty', 'Baby_Products', 'Video_Games']
output_dir = 'data/amazon_review/processed/corpus'
os.makedirs(output_dir, exist_ok=True)

raw_meta_data_dir = 'data/amazon_review/raw'

for domain_name in domain_name_list:

    asin2meta = {}

    with open(f"{raw_meta_data_dir}/{domain_name}/meta_{domain_name}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            asin = item["parent_asin"]
            asin2meta[asin] = item
    
    # Load the file
    file_path = f"{file_dir}/{domain_name}/{domain_name}.data_maps"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse the JSON content
    data = json.loads(content)

    # Extract item info
    id2item = data["id2item"]  # list
    id2meta = data["id2meta"]  # dict
    
    assert len(id2item) == len(id2meta)

    results = []

    for item_numeric_id_str, metadata in id2meta.items():
        try:
            item_index = int(item_numeric_id_str)
            item_id = id2item[item_index]
            results.append({
                "item_id": item_id,
                "category": asin2meta.get(item_id, {}).get("main_category", None),
                "metadata": metadata,
                'title': asin2meta.get(item_id, {}).get("title", None),
            })
        except (ValueError, IndexError):
            continue  # Skip malformed or out-of-range entries
    
    # remove [PAD]
    results = [item for item in results if item["item_id"] != "[PAD]"]

    # Save to JSONL
    output_path_filtered = f"{output_dir}/{domain_name}_items_filtered.jsonl"
    with open(output_path_filtered, "w", encoding="utf-8") as f:
        for item in results:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")
