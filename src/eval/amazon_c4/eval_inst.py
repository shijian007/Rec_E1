import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import os
import pdb

CACHE_DIR = "/srv/local/data/linjc/hub"

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=CACHE_DIR, device_map='auto', torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.eval()
    return tokenizer, model


def evaluate_model(model, tokenizer, data_path, device, model_name, save_dir, batch_size=2, process_id=None, num_processes=None):
    df = pd.read_parquet(data_path)
    
    if process_id is not None:
        # split the data into num_processes parts
        df = df.iloc[process_id::num_processes]
    inputs = [item[0]['content'] for item in df['prompt'].tolist()]
    targets = [item for item in df['item_id'].tolist()]

    model.to(device)
    generated_texts = {}

    save_file_name = f"eval_results_{model_name}_{process_id}.json" if process_id is not None else f"eval_results_{model_name}.json"

    if os.path.exists(os.path.join(save_dir, save_file_name)):
        with open(os.path.join(save_dir, save_file_name), "r") as f:
            generated_texts = json.load(f)

    for batch_start in tqdm(range(0, len(inputs), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(inputs))
        batch_inputs = inputs[batch_start:batch_end]

        # remove already generated texts from the batch_inputs
        # the idx is from batch_start to batch_end
        for idx in range(batch_start, batch_end):
            if str(idx) in generated_texts:
                batch_inputs[idx - batch_start] = ""

        batch_inputs = [item for item in batch_inputs if item != ""]
        
        if len(batch_inputs) == 0:
            continue
        
        tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            # try:
                output_ids = model.generate(**tokenized_inputs, max_new_tokens=256)
            # except:
                # continue
        
        for i, output in enumerate(output_ids):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            idx = batch_start + i
            generated_texts[str(idx)] = {
                "generated_text": generated_text,
                "target": targets[idx]
            }
        
        with open(os.path.join(save_dir, save_file_name), "w") as f:
            json.dump(generated_texts, f, indent=4)
    
    with open(os.path.join(save_dir, save_file_name), "w") as f:
        json.dump(generated_texts, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/Panacea-Zero/matching-qwen2.5-3b-inst-ppo-2gpus/actor/global_step_400")
    parser.add_argument("--data_path", type=str, default="data/matching/qwen-instruct/test.parquet")
    parser.add_argument("--model_name", type=str, default="matching-qwen2.5-3b-inst-ppo-2gpus")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--process_id', type=int, default=None)
    parser.add_argument('--num_processes', type=int, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model_path)
    evaluate_model(model, tokenizer, args.data_path, device, args.model_name, args.save_dir, args.batch_size, args.process_id, args.num_processes)

if __name__ == "__main__":
    main()
