from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode_name', type=str, default='with_reason_no_json')
args = parser.parse_args()

print(args.mode_name)

MODE = args.mode_name

if MODE == 'no_reason':
    train_file_path = 'data/esci/inst/sparse/sft/no_reason/train.parquet'
    val_file_path = 'data/esci/inst/sparse/sft/no_reason/val.parquet'
    output_dir = "./checkpoints/qwen-sft-full-no_reason"
elif MODE == 'no_reason_no_json':
    train_file_path = 'data/esci/inst/sparse/sft/no_json/train.parquet'
    val_file_path = 'data/esci/inst/sparse/sft/no_json/val.parquet'
    output_dir = "./checkpoints/qwen-sft-full-no_reason_no_json"
elif MODE == 'normal':
    train_file_path = 'data/esci/inst/sparse/sft/train.parquet'
    val_file_path = 'data/esci/inst/sparse/sft/val.parquet'
    output_dir = "./checkpoints/qwen-sft-full"
elif MODE == 'with_reason_no_json':
    train_file_path = 'data/esci/inst/sparse/sft/with_reason_no_json/train.parquet'
    val_file_path = 'data/esci/inst/sparse/sft/with_reason_no_json/val.parquet'
    output_dir = "./checkpoints/qwen-sft-full-with_reason_no_json"

model_name = "Qwen/Qwen2.5-3B-Instruct"
cache_dir = "/srv/local/data/linjc/hub"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=cache_dir,
    attn_implementation="flash_attention_2"
)

dataset = load_dataset("parquet", data_files={
    "train": train_file_path,
    "validation": val_file_path
})

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)

# 初始化 SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    args=training_args,
    max_seq_length=1024
)

trainer.train()
