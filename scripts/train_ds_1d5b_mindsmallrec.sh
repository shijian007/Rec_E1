# pyserini 必须安装java21，非java17，指令：apt-get install -y openjdk-21-jdk
# model output path: /opt/data2/xiezhijun/code/Rec-R1/checkpoints/Rec-R1/rec-qwen2.5-3b-inst-grpo-amazon_review/actor/global_step_800

export N_GPUS=2
export BASE_MODEL=/opt/dataclean2/models/deepseek_models/DeepSeek-R1-Distill-Qwen-1.5B
export DATA_DIR=/opt/data2/xiezhijun/code/Rec-R1/database/rl_rec/mind_small_v1d1_153k_0423
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=mindrec-ds-1d5b-grpo-153k_0423_v1d1
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY="3d0dcbfee0162a77806c9cf933dd0a5e8030aa55"
export CUDA_VISIBLE_DEVICES=4,5

DATE=$(date '+%Y-%m-%d-%H-%M-%S')

python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/dev.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=3000 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.optim.lr=1e-5 \
    critic.model.path=$BASE_MODEL \
    critic.ppo_micro_batch_size=2 \
    critic.model.enable_gradient_checkpointing=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['wandb'] \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.project_name=Rec-R1 \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=5 2>&1 | tee exp_log/1d5b-ppo-verl_demo_$DATE.log
