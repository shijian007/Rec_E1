import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import os
import pdb
import re
import numpy as np

CACHE_DIR = "/srv/local/data/linjc/hub"

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map='auto',)
    model.eval()
    return tokenizer, model


def evaluate_model(model, tokenizer, data_path, device, model_name, save_dir, domain_name, batch_size=8):
    df = pd.read_parquet(data_path)

    # only keep those ['data_source] contains domain_name, keep index unchanged
    df = df[df['data_source'].str.contains(domain_name, case=False, na=False)]

    inputs = [item[0]['content'] for item in df['prompt'].tolist()]
    targets = df['item_id'].tolist()
    qids = df.index.tolist()
    
    # if targets[i] is a array, then convert to list
    for i in range(len(targets)):
        # if is a np.ndarray, then convert to list
        if isinstance(targets[i], np.ndarray):
            targets[i] = targets[i].tolist()
    
    model.to(device)
    generated_texts = {}
    
    for batch_start in tqdm(range(0, len(inputs), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(inputs))
        batch_inputs = inputs[batch_start:batch_end]
        
        tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            # try:
                output_ids = model.generate(**tokenized_inputs, max_new_tokens=1024, do_sample=False)
            # except:
                # continue
        
        for i, output in enumerate(output_ids):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            # extract from \nAssistant:
            generated_text = generated_text.split("\nassistant\n")[1]
            
            # findall <answer> </answer>
            answer_pattern = r'<answer>(.*?)</answer>'
            matches = re.findall(answer_pattern, generated_text, re.DOTALL)  # Use re.DOTALL to match multiline content

            if len(matches) > 0:
                generated_text = matches[0]
                try:
                    # json.loads(generated_text)
                    generated_text = json.loads(generated_text)['query']
                except:
                    generated_text = matches[0]

            idx = batch_start + i
            generated_texts[qids[idx]] = {
                "generated_text": generated_text,
                "target": str(targets[idx])
            }
        
        with open(os.path.join(save_dir, f"{model_name}_{domain_name}.json"), "w") as f:
            json.dump(generated_texts, f, indent=4)
    
    with open(os.path.join(save_dir, f"{model_name}_{domain_name}.json"), "w") as f:
        json.dump(generated_texts, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', type=str, choices=['Video_Games', 'Baby', 'Office', 'Sports'], default='Video_Games')
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/amazon_c4/inst/dense/subset_other/test.parquet")
    parser.add_argument("--model_name", type=str, default="Qwen-inst-amazon-c4")
    parser.add_argument("--save_dir", type=str, default="results_dense")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model_path)
    evaluate_model(model, tokenizer, args.data_path, device, args.model_name, args.save_dir, args.domain_name, args.batch_size)

if __name__ == "__main__":
    main()
