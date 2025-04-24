import json


def load_item_metadata_dict(meta_data_path):
    item_metadata_dict = {}
    with open(meta_data_path, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line.strip())
            item_metadata_dict[item["item_id"]] = item['metadata']
    
    return item_metadata_dict