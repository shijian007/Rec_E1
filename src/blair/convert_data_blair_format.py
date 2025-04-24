import os
import json

def convert_json_to_tab_separated(data):
    result = []
    for entry in data:
        user_id = entry["user_id"]
        history_list = entry["history_list"]
        target = entry["target"]

        user_token = f"{user_id}"
        history_token_seq = f"{' '.join(history_list)}"
        target_token = f"{target}"

        line = f"{user_token}\t{history_token_seq}\t{target_token}"
        result.append(line)

    return result


if __name__ == '__main__':
    domain_list = ['All_Beauty', 'Baby_Products', 'Video_Games']

    processed_data_dir = 'data/amazon_review/processed'
    output_data_dir = 'data/amazon_review/processed/new'
    split_data_dir = 'data/amazon_review/ori_split_files'

    for domain in domain_list:
        save_dir = os.path.join(output_data_dir, domain)
        os.makedirs(save_dir, exist_ok=True)


        for split in ['val', 'test']:
            ori_split_data_path = os.path.join(split_data_dir, f"{domain}", f"{split}.json")
            with open(ori_split_data_path, 'r') as f:
                data = json.load(f)
            
            if split == 'val':
                split = 'valid'

            new_data = convert_json_to_tab_separated(data)

            # write head head user_id:token	item_id_list:token_seq	item_id:token
            with open(os.path.join(save_dir, f"{domain}.{split}.inter"), 'w') as f:
                f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
                f.write('\n'.join(new_data))
