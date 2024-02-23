# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava.train.train import preprocess, preprocess_multimodal
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image
from projects.distill.distill_llava_llama import DistillModel

local_rank = None


replace_llama_attn_with_flash_attn()
from transformers.models.llama.modeling_llama import LlamaModel # L708

from llava.train.train import rank0_print, DataArguments
from llava.train.train import find_all_linear_names, smart_tokenizer_and_embedding_resize
from llava.train.train import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    tune_vision_tower: bool = field(default=False)
    tune_vit_from_layer: Optional[int] = field(default=-1)
    tune_entire_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # additional variables
    model_variant: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    select_k: int = 16
    align_logits: bool = False
    align_logits_all: bool = False
    align_image_tokens: bool = False
    align_affinity: bool = False
    mse_distill: bool = False
    align_hidden_embeds: bool = False
    align_attn_map: bool = False
    align_vision_tower: bool = False

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(
        key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return


def get_vision_tower_state_maybe_zero_3(named_params, keys_to_match=['']):
    to_return = {k: t for k, t in named_params if any(
        key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return



class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            stu_processor = self.data_args.stu_image_processor
            tea_processor = self.data_args.tea_image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in stu_processor.image_mean))
                stu_image = stu_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                tea_image = tea_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                stu_image = stu_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                tea_image = tea_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['stu_image'] = stu_image
            data_dict['tea_image'] = tea_image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            stu_crop_size = self.data_args.stu_image_processor.crop_size
            tea_crop_size = self.data_args.tea_image_processor.crop_size
            data_dict['stu_image'] = torch.zeros(3, stu_crop_size['height'], stu_crop_size['width'])
            data_dict['tea_image'] = torch.zeros(3, tea_crop_size['height'], tea_crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'stu_image' in instances[0]:
            stu_images = [instance['stu_image'] for instance in instances]
            if all(x is not None and x.shape == stu_images[0].shape for x in stu_images):
                batch['stu_images'] = torch.stack(stu_images)
            else:
                batch['stu_images'] = stu_images

        if 'tea_image' in instances[0]:
            tea_images = [instance['tea_image'] for instance in instances]
            if all(x is not None and x.shape == tea_images[0].shape for x in tea_images):
                batch['tea_images'] = torch.stack(tea_images)
            else:
                batch['tea_images'] = tea_images
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def unlock_vit(training_args, model_args, vision_tower):
    # lr_of_vit = training_args.vision_tower_lr if training_args.vision_tower_lr is not None and training_args.vision_tower_lr != 0 else training_args.learning_rate

    # rank0_print(f'Tune the vision tower! LR for ViT is {lr_of_vit}.')
    if model_args.tune_vit_from_layer != -1:
        rank0_print(f'Tune the vision tower from layer {model_args.tune_vit_from_layer}!')
    for n, p in vision_tower.named_parameters():
        if model_args.tune_vit_from_layer != -1:
            if 'vision_tower.vision_model.encoder.layers.' in n:
                layer_id = int(
                    n.split('vision_tower.vision_model.encoder.layers.')[-1].split('.')[0])
                if layer_id >= model_args.tune_vit_from_layer:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            else:
                p.requires_grad = False
        else:
            p.requires_grad = True

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif model_args.model_variant is not None and model_args.model_variant == 'mobilevlm':
            from projects.ext.mobilevlm.model.mobilellama import MobileLlamaForCausalLM
            model = MobileLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=compute_dtype, device=training_args.device)

        # if training_args.align_vision_tower:
        #     vision_tower.vision_tower.vision_model.embeddings.patch_embedding = torch.nn.Conv2d(3,768,14,14,bias=False)
        #     vision_tower.vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(577, 768)
        #     vision_tower.vision_tower.vision_model.embeddings.position_ids = torch.arange(577).expand((1, -1))
        #     vision_tower.to(dtype=compute_dtype, device=training_args.device)
        #     vision_tower.requires_grad_(True)

        data_args.stu_image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.tune_vision_tower = training_args.tune_vision_tower = model_args.tune_vision_tower
        model.config.tune_entire_model = training_args.tune_entire_model = model_args.tune_entire_model
        if model_args.tune_entire_model:
            if training_args.lora_enable:
                unlock_vit(training_args, model_args, vision_tower)
            else:
                model.requires_grad_(True)
                unlock_vit(training_args, model_args, vision_tower)
        else:
            if model_args.tune_vision_tower:
                unlock_vit(training_args, model_args, vision_tower)

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.vision_tower_lr = training_args.vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    teacher_model = LlavaLlamaForCausalLM.from_pretrained(
        "liuhaotian/llava-v1.5-13b",
    )

    teacher_model.config.use_cache = False
    teacher_model.config.image_aspect_ratio = data_args.image_aspect_ratio
    teacher_model.config.tokenizer_padding_side = tokenizer.padding_side
    teacher_model.config.tokenizer_model_max_length = tokenizer.model_max_length

    teacher_model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    teacher_model.config.mm_projector_lr = training_args.mm_projector_lr
    teacher_model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

    teacher_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "liuhaotian/llava-v1.5-13b",
        model_max_length=training_args.model_max_length,
        use_fast=False,
        padding_side="right"
    )
    teacher_vision_tower = teacher_model.get_vision_tower()
    if not teacher_vision_tower.is_loaded:
        teacher_vision_tower.load_model()
    data_args.tea_image_processor = teacher_vision_tower.image_processor
    teacher_model.to(device=training_args.device, dtype=compute_dtype)
    teacher_model.requires_grad_(False)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)


    # from llava.train.llava_trainer import LengthGroupedSampler
    # from torch.utils.data import DataLoader
    # train_dataset = data_module['train_dataset']
    # dataloader_params = dict(
    #     sampler = LengthGroupedSampler(batch_size=8,world_size=1,
    #             lengths = train_dataset.modality_lengths,
    #             group_by_modality=True),
    #     batch_size=training_args.per_device_train_batch_size,
    #     collate_fn=data_module['data_collator'],
    #     num_workers=training_args.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    #     worker_init_fn = transformers.trainer_utils.seed_worker
    # )
    # data_loader = DataLoader(train_dataset, **dataloader_params)

    model = DistillModel(
                training_args,
                student_model=model,
                student_tokenizer=tokenizer,
                teacher_model=teacher_model,
                teacher_tokenizer=teacher_tokenizer,
            )
    if training_args.align_hidden_embeds:
        model.embed_projector.requires_grad_(True)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            if model_args.tune_entire_model:
                if trainer.deepspeed:
                    torch.cuda.synchronize()
                trainer.model.get_vision_tower().image_processor.save_pretrained(
                    os.path.join(training_args.output_dir, 'vision_tower'))
                trainer.model.get_vision_tower().vision_tower.vision_model.config.save_pretrained(
                    os.path.join(training_args.output_dir, 'vision_tower'))
                weight_to_save = get_vision_tower_state_maybe_zero_3(
                    trainer.model.get_vision_tower().vision_tower.named_parameters())
                torch.save(weight_to_save, os.path.join(
                    training_args.output_dir, 'vision_tower/pytorch_model.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.student_model.named_parameters(), keys_to_match)
        trainer.model.student_model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

        if getattr(trainer.args, "tune_vision_tower", False):
            if trainer.deepspeed:
                torch.cuda.synchronize()
            trainer.model.student_model.get_vision_tower().image_processor.save_pretrained(
                os.path.join(output_dir, 'vision_tower'))
            trainer.model.student_model.get_vision_tower().vision_tower.vision_model.config.save_pretrained(
                os.path.join(output_dir, 'vision_tower'))
            weight_to_save = get_vision_tower_state_maybe_zero_3(
                trainer.model.student_model.get_vision_tower().vision_tower.named_parameters())
            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                torch.save(weight_to_save, os.path.join(
                    output_dir, 'vision_tower/pytorch_model.bin'))
        return

    if getattr(trainer.args, "tune_vision_tower", False) or getattr(trainer.args, "tune_entire_model", False):
        if trainer.deepspeed:
            torch.cuda.synchronize()
        trainer.model.student_model.get_vision_tower().image_processor.save_pretrained(
            os.path.join(output_dir, 'vision_tower'))
        trainer.model.student_model.get_vision_tower().vision_tower.vision_model.config.save_pretrained(
            os.path.join(output_dir, 'vision_tower'))
        weight_to_save = get_vision_tower_state_maybe_zero_3(
            trainer.model.student_model.get_vision_tower().vision_tower.named_parameters())
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            torch.save(weight_to_save, os.path.join(
                output_dir, 'vision_tower/pytorch_model.bin'))

    if trainer.deepspeed:
        torch.cuda.synchronize()
        if getattr(trainer.model.student_model.model, 'vision_tower', None) is not None:
            del trainer.model.student_model.model.vision_tower
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.student_model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == "__main__":
    train()
