#!/bin/bash
MODEL_PATH=$1

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=. srun -p ${PARTITION} --quotatype=spot --job-name=${JOB_NAME} --gres=gpu:8 --ntasks-per-node=8 --nodes=1 \
python llava/eval/model_vqa_loader_old.py \
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
