#!/bin/bash
PYTHONPATH='.' \
srun -p s1_mm_research --ntasks-per-node=1 --gres=gpu:8 --cpus-per-task=10 --nodes=1 \
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path checkpoints/vicuna-13b-v1.5 \
    --version plain \
    --data_path datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder datasets/LLaVA-Pretrain/images \
    --vision_tower convnext_large_d_320/laion2b_s29b_b131k_ft_soup \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./output/llava-v1.5-13b-pretrain_convnext_l \
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
    --lazy_preprocess True
