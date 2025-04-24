import json
import re
import os
import pdb

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt-35')
parser.add_argument('--dataset_name', type=str, default='scifact')
parser.add_argument('--save_dir', type=str, default='../results/')
args = parser.parse_args()

dataset_name = args.dataset_name

file_path = os.path.join(args.save_dir, f'{args.model_name}_{dataset_name}.json')
with open(file_path, 'r') as file:
    data = json.load(file)


answers = {}
for key, value in data.items():
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", value["generated_text"], re.DOTALL)
    try:
        query_text = match.group(1)
        # json loads
        try:
            gen_query = json.loads(query_text)['query']   
        except:
            gen_query = query_text
        answers[key] = {
            "generated_text": gen_query,
            "target": value["target"],
        }
    except:
        answers[key] = {
            "generated_text": value["generated_text"],
            "target": value["target"],
        }


# save to filename {args.model_name}_postprocessed_scifact.json
file_path = os.path.join(args.save_dir, f'{args.model_name}_post_{dataset_name}.json')
with open(file_path, 'w') as file:
    json.dump(answers, file, indent=4)

