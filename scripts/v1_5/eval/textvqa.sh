#!/bin/bash
MODEL_PATH=$1

PYTHONPATH='.' srun -p s1_mm_dev --quotatype=auto --gres=gpu:8 --ntasks-per-node=1 --nodes=1 \
deepspeed llava/eval/model_vqa_loader.py \
    --model-path ${MODEL_PATH} \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./datasets/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl

rm -rf ./playground/data/eval/textvqa/answers
