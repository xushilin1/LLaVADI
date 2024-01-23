#!/bin/bash
MODEL_PATH=$1

temp_dir=$(mktemp -d)
res_folder=/mnt/petrelfs/xushilin/${temp_dir}
mkdir -p ${res_folder}
PYTHONPATH='.' srun -p s1_mm_research --gres=gpu:8 --ntasks-per-node=1 --nodes=1 \
deepspeed llava/eval/model_vqa_loader.py \
    --model-path ${MODEL_PATH} \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./datasets/coco/val2014 \
    --answers-file ${res_folder}/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ${res_folder}/llava-v1.5-13b.jsonl

rm -rf ${res_folder}