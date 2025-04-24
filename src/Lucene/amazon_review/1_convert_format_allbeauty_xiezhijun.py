import json
import os

def convert_jsonl_for_pyserini(input_file, output_file):
    """Convert JSONL data to Pyserini-compatible format with a structured 'contents' field"""
    docs = []
    all_itemid_set = set()

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            itemid = data["parent_asin"]
            if itemid not in all_itemid_set:
                all_itemid_set.add(itemid)
            else:
                continue
            doc = {
                "id": itemid,  # Unique identifier for search results
                "contents": data['title'],  # Required field for Pyserini
            }
            docs.append(json.dumps(doc))

    print(f"all itemid count: {len(docs)}")
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc + "\n")

    print(f"✅ Converted JSONL saved to {output_file}")




inputfiles = f"/opt/data2/xiezhijun/code/Rec-R1/data/amazon_beauty/meta_All_Beauty.jsonl"
output_file = f"/opt/data2/xiezhijun/code/Rec-R1/database/amazon_review/All_Beauty/jsonl_docs/pyserini.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Example Usage
convert_jsonl_for_pyserini(inputfiles, output_file)
