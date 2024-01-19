#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path output/llava-v1.5-13b \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./datasets/vizwiz/test \
    --answers-file ./datasets/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./datasets/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --result-upload-file ./datasets/eval/vizwiz/answers_upload/llava-v1.5-13b.json
