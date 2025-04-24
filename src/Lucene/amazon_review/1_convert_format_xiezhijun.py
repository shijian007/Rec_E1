import json
import os

def convert_jsonl_for_pyserini(input_file, output_file):
    """Convert JSONL data to Pyserini-compatible format with a structured 'contents' field"""
    docs = []
    all_itemid_set = set()

    for ifile in input_file:
        with open(ifile, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                data = item['history']
                for d in data:
                    itemid = d["asin"]
                    if itemid not in all_itemid_set:
                        all_itemid_set.add(itemid)
                    else:
                        continue
                    doc = {
                        "id": itemid,  # Unique identifier for search results
                        "title": d['title'],
                        "contents": d['text'],  # Required field for Pyserini
                    }
                    docs.append(json.dumps(doc))

    print(f"all itemid count: {len(docs)}")
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc + "\n")

    print(f"✅ Converted JSONL saved to {output_file}")




inputfiles = [f"/opt/data2/xiezhijun/code/Rec-R1/data/amazon_review/split/Beauty/train.jsonl",
             f"/opt/data2/xiezhijun/code/Rec-R1/data/amazon_review/split/Beauty/test.jsonl"]
output_file = f"/opt/data2/xiezhijun/code/Rec-R1/database/amazon_review/Beauty/jsonl_docs/pyserini.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Example Usage
convert_jsonl_for_pyserini(inputfiles, output_file)
