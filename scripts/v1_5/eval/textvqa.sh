#!/bin/bash
MODEL_PATH=$1

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=. srun -p ${PARTITION} --quotatype=spot --job-name=${JOB_NAME} --gres=gpu:1 --ntasks-per-node=1 --nodes=1 \
deepspeed llava/eval/model_vqa_loader.py \
    --model-path ${MODEL_PATH} \
    --question-file ./data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./data/LLaVA-Instruct/textvqa/train_images \
    --answers-file ./data/eval/textvqa/answers/dummy.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./data/eval/textvqa/answers/dummy.jsonl

rm -rf ./data/eval/textvqa/answers
