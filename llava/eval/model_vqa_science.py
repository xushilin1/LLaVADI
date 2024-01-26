import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from PIL import Image
import math
import deepspeed
import pickle

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

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
    model, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=ds_config
    )

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, world_size, rank)
    
    result_part = []
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            if getattr(model.model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # prompt for answer
        if args.answer_prompter:
            outputs_reasoning = outputs
            input_ids = tokenizer_image_token(prompt + outputs_reasoning + ' ###\nANSWER:', tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=64,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            outputs = outputs_reasoning + '\n The answer is ' + outputs

        ans_id = shortuuid.uuid()
        result_part.append({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}})

    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')

    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor) # 每个进程占据一个位置
    
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor

    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        # ordered_results = ordered_results[:len(dataset)]
        answers_file = os.path.expanduser(args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")

        for res in ordered_results:
            ans_file.write(json.dumps(res) + '\n')
       
        ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
