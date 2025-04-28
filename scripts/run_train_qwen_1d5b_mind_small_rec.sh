set -x


export N_GPUS=2
export BASE_MODEL=/opt/dataclean2/models/qwen_models/Qwen2.5-1.5B-Instruct
export DATA_DIR=/opt/data2/xiezhijun/code/Rec-R1/database/rl_rec/mindsmall_v2_153k_0427
export ROLLOUT_TP_SIZE=2
export PROJECT_NAME=Rec-E1
export EXPERIMENT_NAME=mindsmall-qwen-1d5b-grpo-v2_153k_0427
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY="3d0dcbfee0162a77806c9cf933dd0a5e8030aa55"
export CUDA_VISIBLE_DEVICES=4,5

DATE=$(date '+%Y-%m-%d-%H-%M-%S')


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/dev.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=4000 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 $@