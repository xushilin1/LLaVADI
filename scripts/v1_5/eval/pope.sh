#!/bin/bash
PYTHONPATH='.' CUDA_VISIBLE_DEVICES=0,1 \
deepspeed llava/eval/model_vqa_loader.py \
    --model-path checkpoints/llava-v1.5-13b \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./datasets/coco/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl
