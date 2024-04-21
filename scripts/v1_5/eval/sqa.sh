#!/bin/bash
MODEL_PATH=$1

PYTHONPATH='.' srun -p s1_mm_dev --quotatype=auto --gres=gpu:8 --ntasks-per-node=1 --nodes=1 \
deepspeed  llava/eval/model_vqa_science.py \
    --model-path ${MODEL_PATH} \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./datasets/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_result.json

rm -rf ./playground/data/eval/scienceqa/answers/