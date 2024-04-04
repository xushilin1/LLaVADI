#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=1
export MASTER_PORT=29501
export CPUS_PER_TASK=32
export QUOTA=auto

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p s1_mm_research \
    --nodes=$NNODES \
    --ntasks-per-node=1 \
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA} \
    bash -c 'torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID \
    --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) \
    --master_port ${MASTER_PORT} llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path checkpoints/MobileLLaMA-2.7B-Chat \
    --version plain \
    --data_path datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder datasets/LLaVA-Pretrain/images \
    --vision_tower checkpoints/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir gg \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True'
