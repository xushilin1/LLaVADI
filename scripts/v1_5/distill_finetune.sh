#!/bin/bash
set -x

wandb login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=2
export MASTER_PORT=29501
export CPUS_PER_TASK=32
export QUOTA=auto

SRUN_ARGS=${SRUN_ARGS:-""}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p s1_mm_research \
    --nodes=$NNODES \
    --ntasks-per-node=1 \
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA} \
    ${SRUN_ARGS} \
    bash -c 'torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} \
    projects/distill/distill_train.py \
    --teacher_model_path output/finetune/llava_13B_2M \
    --align_logits True \
    --align_hidden_embeds True \
    --align_on_policy False \
    --align_contrastive_affinity False \
    --tune_entire_model False \
    --tune_vit_from_layer 6 \
    --model_name_or_path output/finetune/llava_MobileLLaMA-2.7B-2M/ \
    --deepspeed ./scripts/zero3.json \
    --version v1 \
    --data_path datasets/MobileVLM_V2_FT_Mix2M/MobileVLM_V2_FT_Mix2M.json \
    --image_folder ./datasets \
    --vision_tower checkpoints/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir output/llava_mobile_llama_3b_2M_stu_init_tea_init \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True'