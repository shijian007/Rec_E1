import argparse
import json
import glob
import os
import pdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', type=str, default='Baby')
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    
    if args.split == 'train':
        # read all the files in data/amazon_c4/filtered/raw
        json_files = glob.glob(f"data/amazon_c4/gen_train/raw/{args.domain_name}/*.jsonl")

        all_data = []
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    all_data.append(data)
        
        for i, entry in enumerate(all_data):
            entry["qid"] = i
            entry["ori_rating"] = int(entry["ori_rating"])
        
        output_file = f"data/amazon_c4/subset/{args.domain_name}/train.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
    
    elif args.split == 'test':
        # read data/amazon_c4/raw/test.csv
        # import pandas as pd
        # df = pd.read_csv('data/amazon_c4/raw/test.csv')
        # json_data = df.to_dict(orient="records")

        # output_json_path = "data/amazon_c4/test.json"
        # with open(output_json_path, "w", encoding="utf-8") as f:
        #     json.dump(json_data, f, indent=4, ensure_ascii=False)

        # read data/amazon_c4/test_subset/{args.domain_name}.json
        with open(f"data/amazon_c4/test_subset/{args.domain_name}.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)

        for i, entry in enumerate(test_data):
            entry["ori_rating"] = int(entry["ori_rating"])

        output_file = f"data/amazon_c4/subset/{args.domain_name}/test.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)


