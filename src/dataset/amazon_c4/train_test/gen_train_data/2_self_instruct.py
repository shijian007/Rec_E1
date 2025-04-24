import os
import json
import random
import re
import time
from typing import List, Dict
from difflib import SequenceMatcher
from pathlib import Path
import argparse

import sys
sys.path.append('./')
from src.utils.gpt_azure import gpt_chat_4omini




def load_seed_tasks(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# ==== 构造 prompt ====
def build_prompt(seed_examples: List[Dict[str, str]]) -> str:
    prompt = "You are an instruction generation assistant. Given examples of instructions, inputs, and outputs, generate a new, creative instruction task.\n\n"
    for i, ex in enumerate(seed_examples):
        input_text = ex['instances'][0].get('input', '') or "<noinput>"
        output_text = ex['instances'][0].get('output', '')
        prompt += f"###\nInstruction: {ex['instruction']}\nInput: {input_text}\nOutput: {output_text}\n"
    prompt += "###\nInstruction:"
    return prompt

# ==== 解析返回文本 ====
def extract_instruction_blocks(text: str) -> List[Dict[str, str]]:
    blocks = re.split(r"###", text)
    tasks = []
    for block in blocks:
        instr_match = re.search(r"Instruction:\s*(.+)", block)
        input_match = re.search(r"Input:\s*(.+)", block)
        output_match = re.search(r"Output:\s*(.+)", block)
        if instr_match and output_match:
            tasks.append({
                "instruction": instr_match.group(1).strip(),
                "input": input_match.group(1).strip() if input_match else "",
                "output": output_match.group(1).strip()
            })
    return tasks

# ==== 简单相似度去重 ====
def is_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    return SequenceMatcher(None, a, b).ratio() > threshold

# ==== 保存为 JSONL ====
def save_to_jsonl(data: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ==== 主流程 ====
def generate_self_instruct():
    seed_tasks = load_seed_tasks(SEED_FILE)
    all_instructions = set()
    generated_data = []

    while len(generated_data) < NUM_TO_GENERATE:
        batch_prompts = []
        for _ in range(BATCH_SIZE):
            samples = random.sample(seed_tasks, 3)
            prompt = build_prompt(samples)
            batch_prompts.append(prompt)

        for prompt in batch_prompts:
            try:
                response = query_gpt([
                    {"role": "user", "content": prompt}
                ])
                tasks = extract_instruction_blocks(response)
                for task in tasks:
                    if task["instruction"] and not any(is_similar(task["instruction"], seen) for seen in all_instructions):
                        generated_data.append(task)
                        all_instructions.add(task["instruction"])
                        print(f"✅ New instruction: {task['instruction'][:60]}")
                    else:
                        print("⚠️  Skipped similar or empty instruction.")
            except Exception as e:
                print(f"❌ Error: {e}")
            time.sleep(1)  # 避免触发速率限制

    save_to_jsonl(generated_data, OUTPUT_FILE)
    print(f"\n🎉 Done! Saved {len(generated_data)} tasks to {OUTPUT_FILE}")

if __name__ == "__main__":
    SEED_FILE = "./seed_tasks.jsonl"
    OUTPUT_FILE = "./generated_instructions.jsonl"
    NUM_TO_GENERATE = 50
    BATCH_SIZE = 5
    TEMPERATURE = 0.8
    MODEL_NAME = "gpt-4o" 
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_file", type=str, default='')    
