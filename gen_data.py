import os

import json
import math
import torch
import argparse
import deepspeed
from tqdm import tqdm
from PIL import Image

from llava.model import LlavaLlamaForCausalLM
from transformers import AutoTokenizer
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


image_folder = 'datasets'
temperature = 1
top_p = None
num_beams = 1
max_new_tokens=128


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def main(args):
    # rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    rank = 0
    world_size = 1

    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 100,
        "optimizer": {"type": "AdamW", "params": {"lr": 0.001, "weight_decay": 0.0,
                                                "betas": (0.1, 0.1)}},
        "scheduler": {"type": "WarmupDecayLR",
                    "params": {"total_num_steps": 1, "warmup_min_lr": 0,
                                "warmup_max_lr": 0.0001, "warmup_num_steps": 100, "warmup_type": "linear"}},
        "gradient_clipping": 1.0,
        "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True,
                            "reduce_scatter": True, "reduce_bucket_size": 5e8,
                            "allgather_bucket_size": 5e8}
    }
 
    model_path = 'output/finetune/llava_MobileLLaMA-1.4B-Chat'
    model_path = 'checkpoints/MobileVLM-3B'
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model_config = model.config

    model.to('cuda')
    # model, _, _, _ = deepspeed.initialize(
    #     model=model, model_parameters=model.parameters(), config=ds_config
    # )

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=model.device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    questions = json.load(open('datasets/LLaVA-Instruct-150K/llava_v1_5_mix665k_100.json'))
    questions = split_list(questions, world_size)[rank]
    new_data = []

    for line in tqdm(questions):
        conversations = line['conversations']
        if 'image' in line:
            image_file = line['image']
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image_tensor = process_images([image], image_processor, model_config)
            image_tensor = image_tensor.to(dtype=torch.float16, device=model.device, non_blocking=True)
        else:
            image_tensor = None

        new_conversations = []

        conv = conv_templates['v1'].copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        for i in range(len(conversations)):
            sentence = conversations[i]
            role = sentence["from"]
            if role == 'gpt':
                continue
            assert role == 'human'
            new_conversations.append({'from': 'human', 'value': sentence['value']})
            conv.append_message("USER", sentence['value'])
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt + "ASSISTANT:", tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids = input_ids.unsqueeze(0).to(model.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            conv.append_message('ASSISTANT', outputs)
            new_conversations.append({'from': 'gpt', 'value': outputs})

        line['conversations'] = new_conversations
        new_data.append(line)
    
        with open(f'{rank}.json', 'w', encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    main(args)