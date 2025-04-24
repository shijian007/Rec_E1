# MODEL_NAME=checkpoints/qwen-sft-full-no_reason/checkpoint-562
# MODEL_NAME=checkpoints/qwen-sft-full-no_reason_no_json/checkpoint-562
MODEL_NAME=checkpoints/qwen-sft-full-with_reason_no_json/checkpoint-562

echo $MODEL_NAME

lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME \
    --tasks mmlu \
    --device cuda:1 \
    --batch_size 16


# --model_args pretrained=checkpoints/Rec-R1-esci/esci-qwen2.5-3b-inst-grpo-2gpus/actor/global_step_1400 \

# Qwen/Qwen2.5-3B-Instruct

# ifeval