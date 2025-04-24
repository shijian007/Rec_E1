import json
import os

# def convert(input_file, output_file, split):
#     user_data = {}

#     with open(input_file, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     data_count_needed_dict = {
#         'train': 10000000,
#         'val': 1000000000,
#         'test': 1000000000
#     }

#     for line in lines[1:]:
#         parts = line.strip().split("\t")
#         if len(parts) != 3:
#             continue
#         user_id = parts[0]
#         item_id_list = parts[1].split()
#         item_id = parts[2]

#         # Keep the entry with the longest history per user
#         if user_id not in user_data or len(item_id_list) > len(user_data[user_id]["history_list"]):
#             user_data[user_id] = {
#                 "user_id": user_id,
#                 "history_list": item_id_list,
#                 "target": item_id
#             }

#     # Limit to the number of records needed
#     data = list(user_data.values())[:data_count_needed_dict[split]]

#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
    
#     print(f"Conversion completed. {len(data)} records written to {output_file}")

def convert(input_file, output_file):
    data = []

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        user_id = parts[0]
        item_id_list = parts[1].split()
        item_id = parts[2]

        record = {
            "user_id": user_id,
            "history_list": item_id_list,
            "target": item_id
        }
        data.append(record)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Conversion completed. {len(data)} records written to {output_file}")


if __name__ == '__main__':
    domain_list = ['All_Beauty']
    setting_list = ['transductive', 'inductive']
    raw_data_dir = 'data/amazon_review/processed_filtered'
    output_dir = 'data/amazon_review/ori_split_files'
    
    for setting in setting_list:
        for domain in domain_list:
            for split in ['train', 'valid', 'test']:
                input_file = os.path.join(raw_data_dir, setting, domain, f'{domain}.{split}.inter')
                actual_split = 'val' if split == 'valid' else split
                output_file = os.path.join(output_dir, setting, domain, f'{actual_split}.json')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                convert(input_file, output_file)