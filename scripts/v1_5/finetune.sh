#!/bin/bash
PYTHONPATH='.' \
srun -p s1_mm_research --ntasks-per-node=1 --gres=gpu:8 --cpus-per-task=10 --nodes=1 \
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path checkpoints/vicuna-13b-v1.5 \
    --version v1 \
    --data_path datasets/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
    --image_folder ./datasets \
    --vision_tower checkpoints/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./output/llava-v1.5-13b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./output/llava-v1.5-13b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
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
    --lazy_preprocess True
