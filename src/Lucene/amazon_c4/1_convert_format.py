import json
import os

def convert_jsonl_for_pyserini(input_file, output_file):
    """Convert JSONL data to Pyserini-compatible format with a structured 'contents' field"""
    docs = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())

            # Create JSON document with a clear structure
            doc = {
                "id": data["item_id"],  
                "contents": data['metadata'].strip()
            }

            docs.append(json.dumps(doc))

    with open(output_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc + "\n")
    
    print(f"✅ Converted JSONL saved to {output_file}")


ori_data_dir = "data/amazon_c4/raw/sampled_item_metadata_1M.jsonl"
output_file = "database/amazon_c4/jsonl_docs/amazon_c4_metadata.jsonl"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Example Usage
convert_jsonl_for_pyserini(ori_data_dir, output_file)
