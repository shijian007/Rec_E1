import pandas as pd
import os
import argparse
from tqdm import tqdm
import sys
import pdb
import json
import numpy as np

tqdm.pandas()

sys.path.append('./')

from src.utils.claude_aws import chat_sonnet, chat_haiku



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='claude-3.5')
    parser.add_argument('--save_dir', type=str, default='../results')
    parser.add_argument("--data_path", type=str, default="data/local_index_search/scifact/sparse/test.parquet")
    parser.add_argument('--dataset', type=str, default='scifact')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    

    df = pd.read_parquet(args.data_path)

    inputs = [item[0]['content'] for item in df['prompt'].tolist()]
    try:
        targets = df['target'].tolist()
    except:
        targets = df['label'].tolist()
    try:
        qids = df['qid'].tolist()
    except:
        qids = list(range(len(df)))

    # if targets[i] is a array, then convert to list
    for i in range(len(targets)):
        # if is a np.ndarray, then convert to list
        if isinstance(targets[i], np.ndarray):
            targets[i] = targets[i].tolist()
    
    
    output_dict = {}

    # read the existing output file
    if os.path.exists(os.path.join(args.save_dir, f'{args.model_name}_{args.dataset}.json')):
        with open(os.path.join(args.save_dir, f'{args.model_name}_{args.dataset}.json'), 'r') as f:
            output_dict = json.load(f)
    

    i = 0
    for idx, prompt in enumerate(tqdm(inputs)):
        if qids[idx] in output_dict:
            i += 1
            continue
        # extract the prompt from "<|im_start|>user" to <|im_start|>assistant
        prompt = prompt.split("<|im_start|>user", 1)[1]
        prompt = prompt.split("<|im_start|>assistant", 1)[0]

        prompt = prompt.replace("<im_end>", "")
        prompt = prompt.strip()

        prompt = prompt.replace("{", "(")
        prompt = prompt.replace("}", ")")
        attempts = 0
        while attempts < 10:
            try:
                if args.model_name == 'claude-3.5':            
                    decoded = chat_sonnet(prompt)
                elif args.model_name == 'claude-haiku':
                    decoded = chat_haiku(prompt)
                
                break
            except Exception as e:
                print(e)
                attempts += 1
                continue
        
        output_dict[qids[idx]] = {
            "generated_text": decoded,
            "target": str(targets[idx])
        }
        
        if i % 100 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}_{args.dataset}.json'), 'w') as f:
                json.dump(output_dict, f, indent=4)

        i += 1
    
    with open(os.path.join(args.save_dir, f'{args.model_name}_{args.dataset}.json'), 'w') as f:
        json.dump(output_dict, f, indent=4)