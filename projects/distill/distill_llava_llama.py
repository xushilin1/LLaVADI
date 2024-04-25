#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

import torch.nn.functional as F
from llava.train.train import preprocess

class DistillModel(nn.Module):
    

    def __init__(self,
                 args,
                 student_model = None,
                 student_tokenizer = None,
                 teacher_model = None,
                 teacher_tokenizer = None,):
        super(DistillModel, self).__init__()
        self.args = args
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        if self.args.align_hidden_embeds:
            self.embed_projector = nn.Sequential(
                nn.Linear(self.student_model.config.hidden_size, self.teacher_model.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.teacher_model.config.hidden_size, self.teacher_model.config.hidden_size),
            )

    @property
    def config(self):
        return self.student_model.config
    
    @property
    def gradient_checkpointing_enable(self):
        return self.student_model.gradient_checkpointing_enable

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        destination = self.student_model.state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        return destination
        # for param_name, lora_name in self.params_with_lora.items():
        #     destination[prefix + param_name] = eval(f'self.{param_name}').data
        # return destination

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        stu_images: Optional[torch.FloatTensor] = None,
        tea_images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if self.args.align_on_policy and torch.rand(1) < 0.5:
            from llava.constants import DEFAULT_IMAGE_TOKEN
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
            bs_new_input_ids, bs_new_attn_mask, bs_new_target = [], [], []
            temperature = 1.0
            eos_id = self.student_tokenizer.eos_token_id
            try:
                for i in range(len(input_ids)):
                    new_target = []
                    conv = conv_templates['v1'].copy()

                    input_id = input_ids[i].clone()
                    has_img = (input_id == IMAGE_TOKEN_INDEX).sum() > 0

                    image_tensor = None
                    if has_img:
                        image_tensor = stu_images[i:i+1].to(dtype=self.student_model.dtype, device=self.student_model.device, non_blocking=True)
                        input_id[input_id == IMAGE_TOKEN_INDEX] = self.student_tokenizer.pad_token_id
                    
                    conversations = self.student_tokenizer.decode(input_id, skip_special_tokens=True)
                    conversations = conversations.split('USER: ')[1:]
                    source = []
                    for conv_i, sent in enumerate(conversations):
                        qs = sent.split('ASSISTANT')[0].strip()
                        source.append({'from':'human', 'value':qs})
                        if conv_i == 0 and has_img:
                            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                        conv.append_message("USER", qs)
                        prompt = conv.get_prompt()
                        input_ids_copy = tokenizer_image_token(prompt + "ASSISTANT:", self.student_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                        new_input_ids = input_ids_copy.tolist()
                        input_ids_copy = input_ids_copy.unsqueeze(0).to(self.student_model.device)
                        with torch.no_grad():
                            output_ids = self.student_model.generate(
                                input_ids_copy,
                                images=image_tensor,
                                do_sample=True if temperature > 0 else False,
                                temperature=temperature,
                                top_p=None,
                                num_beams=1,
                                max_new_tokens=64,
                                use_cache=False)
                        input_token_len = input_ids_copy.shape[1]
                        n_diff_input_output = (input_ids_copy != output_ids[:, :input_token_len]).sum().item()
                        if n_diff_input_output > 0:
                            raise ValueError(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                        
                        outputs = self.student_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                        outputs = outputs.strip()
                        source.append({'from':'gpt', 'value':outputs})
                        conv.append_message('ASSISTANT', outputs)
                    
                    resp = preprocess([source], self.student_tokenizer, has_img)
                    new_input_ids = resp['input_ids']
                    new_target = resp['labels']
                        
                    if len(new_input_ids) > self.student_tokenizer.model_max_length:
                        new_input_ids = new_input_ids[:self.student_tokenizer.model_max_length]
                        new_target = new_target[:self.student_tokenizer.model_max_length]
                    
                    new_input_ids = torch.tensor(new_input_ids).to(input_ids)
                    attn_mask = torch.ones_like(new_input_ids, dtype=torch.bool)
                    new_target = torch.tensor(new_target).to(labels)

                    bs_new_input_ids.append(new_input_ids)
                    bs_new_attn_mask.append(attn_mask)
                    bs_new_target.append(new_target)

                input_ids = torch.nn.utils.rnn.pad_sequence(
                    bs_new_input_ids,
                    batch_first=True,
                    padding_value=self.student_tokenizer.pad_token_id)
                
                attention_mask = torch.nn.utils.rnn.pad_sequence(
                    bs_new_attn_mask,
                    batch_first=True,
                    padding_value=0)
                
                labels = torch.nn.utils.rnn.pad_sequence(
                    bs_new_target,
                    batch_first=True,
                    padding_value=IGNORE_INDEX)
            
            except Exception as e:
                print(e)
            
        (
            teacher_input_ids,
            teacher_position_ids,
            teacher_attention_mask,
            teacher_past_key_values,
            teacher_inputs_embeds,
            teacher_labels
        ) = self.teacher_model.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            tea_images
        )

        teacher_result = self.teacher_model.forward(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
            position_ids=teacher_position_ids,
            past_key_values=teacher_past_key_values,
            inputs_embeds=teacher_inputs_embeds,
            labels=teacher_labels,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            images=tea_images,
            return_dict=return_dict
        )

        (
            stu_input_ids,
            stu_position_ids,
            stu_attention_mask,
            stu_past_key_values,
            stu_inputs_embeds,
            stu_labels
        ) = self.student_model.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            stu_images
        )

        if self.args.align_image_tokens:
            teacher_input_embeds = teacher_result.hidden_states[0] #(bs, 657, 5120)
            teacher_last_embeds = teacher_result.hidden_states[-1]
            
            new_stu_inputs_embeds = []
            new_stu_attention_mask = []
            new_stu_labels = []

            for batch_idx, cur_input_ids in enumerate(input_ids):
                if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0: # no image in conversation
                    stu_select_img_token = stu_inputs_embeds.new_zeros((self.args.select_k, self.config.hidden_size))
                    stu_select_attention_mask = stu_attention_mask.new_zeros(self.args.select_k)
                    stu_select_labels = stu_labels.new_ones(self.args.select_k) * IGNORE_INDEX
                    image_token_indices = 35
                else:
                    # num_img_token = teacher_input_embeds.shape[1] - cur_input_ids.shape[0] + 1
                    num_img_token = 576  # HACK: hardcode
                    image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].item()

                    image_masks = torch.zeros(teacher_input_embeds.shape[1], dtype=torch.bool)  
                    image_masks[image_token_indices:image_token_indices+num_img_token] = True
                    
                    answer_masks = (teacher_labels[batch_idx] != IGNORE_INDEX)
                    num_ans_token = answer_masks.sum()
                    if num_ans_token == 0:
                        # there are two cases without answer tokens!!!!
                        # data_id: 000000047952, 000000178275
                        stu_select_img_token = stu_inputs_embeds.new_zeros((self.args.select_k, self.config.hidden_size))
                        stu_select_attention_mask = stu_attention_mask.new_zeros(self.args.select_k)
                        stu_select_labels = stu_labels.new_ones(self.args.select_k) * IGNORE_INDEX
                    else:
                        image_embed = teacher_input_embeds[batch_idx][image_masks]
                        answer_embed = teacher_last_embeds[batch_idx][answer_masks]

                        score = torch.matmul(image_embed, answer_embed.T) # (num_img_token, num_ans_token)
                        score = score.softmax(dim=-1)
                        select_idx = score.view(-1).argsort(descending=True)[:self.args.select_k] // num_ans_token
                                
                        # student embeddings
                        stu_select_img_token = stu_inputs_embeds[batch_idx][image_masks][select_idx]
                        stu_select_attention_mask = stu_attention_mask[batch_idx].new_ones(stu_select_img_token.shape[0])
                        stu_select_labels = stu_labels[batch_idx].new_ones(stu_select_img_token.shape[0]) * IGNORE_INDEX
                
                new_stu_inputs_embeds.append(
                    torch.cat([stu_inputs_embeds[batch_idx][:image_token_indices], 
                               stu_select_img_token, 
                               stu_inputs_embeds[batch_idx][image_token_indices:]]
                    )
                )
                new_stu_attention_mask.append(
                    torch.cat([stu_attention_mask[batch_idx][:image_token_indices], 
                               stu_select_attention_mask,
                               stu_attention_mask[batch_idx][image_token_indices:]]
                    )
                )
                new_stu_labels.append(
                    torch.cat([stu_labels[batch_idx][:image_token_indices], 
                               stu_select_labels,
                               stu_labels[batch_idx][image_token_indices:]]
                    )
                )
            stu_inputs_embeds = torch.stack(new_stu_inputs_embeds, dim=0)
            stu_attention_mask = torch.stack(new_stu_attention_mask, dim=0)
            stu_labels = torch.stack(new_stu_labels, dim=0)

        student_result = self.student_model.forward(
            input_ids=stu_input_ids,
            attention_mask=stu_attention_mask,
            position_ids=stu_position_ids,
            past_key_values=stu_past_key_values,
            inputs_embeds=stu_inputs_embeds,
            labels=stu_labels,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            images=stu_images,
            return_dict=return_dict
        )
        
        loss = student_result.loss
        
        if self.args.align_logits:
            distill_loss = 0
            image_masks, answer_masks = self.get_image_masks(input_ids, stu_labels, stu_attention_mask)
            for i in range(labels.shape[0]):
                if answer_masks[i].sum() == 0:
                    continue
                if self.args.align_logits_all:
                    tea_logits = teacher_result.logits[i][stu_attention_mask[i]]
                    stu_logits = student_result.logits[i][stu_attention_mask[i]]
                else:
                    stu_shift_logits = student_result.logits[i]
                    stu_logits = stu_shift_logits[stu_labels[i] != IGNORE_INDEX]
                    
                    teacher_shift_logits = teacher_result.logits[i]
                    tea_logits = teacher_shift_logits[teacher_labels[i] != IGNORE_INDEX]
                    
                    # FIXME: maybe truncate by model_max_length
                    stu_logits = stu_logits[:tea_logits.shape[0]]

                if self.args.norm_logits:
                    def normalize(logit):
                        mean = logit.mean(dim=-1, keepdims=True)
                        stdv = logit.std(dim=-1, keepdims=True)
                        return (logit - mean) / (1e-7 + stdv)
                    stu_logits = normalize(stu_logits)
                    tea_logits = normalize(tea_logits)

                if self.args.mse_distill:
                    distill_loss += F.mse_loss(stu_logits, tea_logits)
                else:
                    if self.args.reverse_kd:
                        distill_loss += F.kl_div(
                            F.log_softmax(tea_logits, dim=-1),
                            F.softmax(stu_logits, dim=-1),
                            reduction='batchmean',
                        )
                    elif self.args.jsd:
                        stu_logits = F.softmax(stu_logits, dim=-1)
                        tea_logits = F.softmax(tea_logits, dim=-1)
                        m = ((stu_logits + tea_logits) / 2).log()
                        distill_loss += (F.kl_div(m, stu_logits, reduction='batchmean') + F.kl_div(m, tea_logits, reduction='batchmean')) / 2
                    else:
                        distill_loss += F.kl_div(
                            F.log_softmax(stu_logits, dim=-1),
                            F.softmax(tea_logits, dim=-1),
                            reduction='batchmean',
                        )
                    # distill_loss += F.cross_entropy(stu_logits, tea_logits.argmax(-1))
            distill_loss /= labels.shape[0]
            loss += distill_loss
        
        if self.args.align_sparse_logits:
            distill_loss = 0
            image_masks, answer_masks = self.get_image_masks(input_ids, stu_labels, stu_attention_mask)
            stu_embeds =  student_result.hidden_states[-1] # TODO: use hidden embedds or logits
            tea_embeds = teacher_result.hidden_states[-1]
            # stu_embeds = student_result.logits
            # tea_embeds = teacher_result.logits
            for i in range(labels.shape[0]):
                img_mask, ans_mask = image_masks[i], answer_masks[i]
                if img_mask.sum() == 0 or ans_mask.sum() == 0:
                    continue
                tea_embed = F.normalize(tea_embeds[i], dim=-1)
                affinity = tea_embed[img_mask] @ tea_embed[ans_mask].T

                index = affinity.view(-1).argsort(descending=True) // affinity.shape[0]
                index = index[:200].unique()

                stu_shift_logits = student_result.logits[i]
                stu_logits = stu_shift_logits[stu_labels[i] != IGNORE_INDEX]
                
                teacher_shift_logits = teacher_result.logits[i]
                teacher_logits = teacher_shift_logits[teacher_labels[i] != IGNORE_INDEX]
                
                # FIXME: maybe truncate by model_max_length
                stu_logits = stu_logits[:teacher_logits.shape[0]]
                
                distill_loss += F.kl_div(
                    F.log_softmax(stu_logits[index] / 0.7, dim=-1),
                    F.softmax(teacher_logits[index] / 0.7, dim=-1),
                    reduction='batchmean',
                ) * 0.7 * 0.7

            distill_loss /= labels.shape[0]
            loss += distill_loss

        if self.args.align_affinity:
            teacher_embeds = torch.stack(teacher_result.hidden_states, dim=1) #(bs, layers, N, 5120)
            student_embeds = torch.stack(student_result.hidden_states, dim=1) #(bs, layers, N, 2048)
            
            # HACK: sample middle layer embeddings uniformly
            # teacher_layer = torch.linspace(0, teacher_embeds.shape[1]-1, steps=10).long()
            # student_layer = torch.linspace(0, student_embeds.shape[1]-1, steps=10).long()
            teacher_layer, student_layer = [-1], [-1]
            
            teacher_embeds = teacher_embeds[:, teacher_layer, :, :]
            student_embeds = student_embeds[:, student_layer, :, :]
            teacher_embeds = F.normalize(teacher_embeds, dim=-1)
            student_embeds = F.normalize(student_embeds, dim=-1)
            teacher_affinity = teacher_embeds @ teacher_embeds.transpose(-1, -2)
            student_affinity = student_embeds @ student_embeds.transpose(-1, -2)

            image_masks, answer_masks = self.get_image_masks(input_ids, stu_labels, stu_attention_mask)
            image_masks = image_masks.unsqueeze(2)
            answer_masks = answer_masks.unsqueeze(1)

            affinity_loss = F.mse_loss(student_affinity, teacher_affinity, reduction='none')
            masks = (image_masks * answer_masks).unsqueeze(1)
            affinity_loss = (affinity_loss * masks).sum() / (masks.sum() + 1e-6)

            loss += affinity_loss

        if self.args.align_contrastive_affinity:
            contrastive_loss = 0
            stu_embeds =  student_result.hidden_states[-1] # (bs, N, D)
            tea_embeds = teacher_result.hidden_states[-1]
            image_masks, answer_masks = self.get_image_masks(input_ids, stu_labels, stu_attention_mask)
            bs = input_ids.shape[0]
            for i in range(bs):
                if image_masks[i].sum() == 0 or answer_masks[i].sum() == 0:
                    continue
                for j in range(bs):
                    if image_masks[j].sum() == 0 or answer_masks[j].sum() == 0:
                        continue
                    stu_img_embed = F.normalize(stu_embeds[i][image_masks[i]], dim=-1) # (num_img, D)
                    tea_img_embed = F.normalize(tea_embeds[i][image_masks[i]], dim=-1)

                    stu_ans_embed = F.normalize(stu_embeds[j][answer_masks[j]], dim=-1) # (num_token, D)
                    tea_ans_embed = F.normalize(tea_embeds[j][answer_masks[j]], dim=-1)

                    stu_img_ans = torch.matmul(stu_img_embed, stu_ans_embed.T)
                    tea_img_ans = torch.matmul(tea_img_embed, tea_ans_embed.T)

                    stu_ans_img = torch.matmul(stu_ans_embed, stu_img_embed.T)
                    tea_ans_img = torch.matmul(tea_ans_embed, tea_img_embed.T)

                    contrastive_loss += (
                        F.cross_entropy(stu_img_ans, tea_img_ans) + 
                        F.cross_entropy(stu_ans_img, tea_ans_img)
                    ) / 2 / answer_masks[j].sum()
            loss += (contrastive_loss / bs / bs)

        if self.args.align_hidden_embeds:
            teacher_embeds = torch.stack(teacher_result.hidden_states, dim=1) #(bs, layers, N, 5120)
            student_embeds = torch.stack(student_result.hidden_states, dim=1) #(bs, layers, N, 2048)
            teacher_layer = [-1]
            student_layer = [-1]
            teacher_embeds = teacher_embeds[:, teacher_layer, :, :] # (bs, 1, N, 5120)
            student_embeds = student_embeds[:, student_layer, :, :]

            # teacher_embeds = self.embed_projector(teacher_embeds)
            student_embeds = self.embed_projector(student_embeds)
            
            # mse_loss = 1 - F.cosine_similarity(student_embeds, teacher_embeds, dim=-1)
            distill_loss = 0
            for i in range(labels.shape[0]):
                stu_mask = (stu_labels[i] != IGNORE_INDEX) & stu_attention_mask[i]
                tea_mask = (teacher_labels[i] != IGNORE_INDEX) & teacher_attention_mask[i]
                if stu_mask.sum() == 0:
                    continue
                stu_embed = student_embeds[i, :, stu_mask]
                tea_embed = teacher_embeds[i, :, tea_mask]

                if stu_embed.shape[1] > tea_embed.shape[1]:
                    stu_embed = stu_embed[:, :tea_embed.shape[1]]
                if tea_embed.shape[1] > stu_embed.shape[1]:
                    tea_embed = tea_embed[:, :stu_embed.shape[1]]

                distill_loss += F.mse_loss(stu_embed, tea_embed)
                
            loss = loss + distill_loss / labels.shape[0]

        if self.args.align_attn_map:
            # NOTE: flash attention will not return attentions
            if student_result.attentions[0] is not None:
                # teacher_embeds = torch.stack(teacher_result.attentions, dim=1) # (bs, num_layer, num_heads, L, L)
                # student_embeds = torch.stack(student_result.attentions, dim=1)
                # teacher_layer = torch.linspace(0, teacher_embeds.shape[1]-1, steps=10).long()
                # student_layer = torch.linspace(0, student_embeds.shape[1]-1, steps=10).long()
            
                # teacher_embeds = teacher_embeds[:, teacher_layer]
                # student_embeds = student_embeds[:, student_layer]
                teacher_embeds = teacher_result.attentions[-1]
                student_embeds = student_result.attentions[-1]
                attn_loss = F.mse_loss(student_embeds, teacher_embeds, reduction='none')

                image_masks, answer_masks = [], []
                for batch_idx, cur_input_ids in enumerate(input_ids):
                    img_mask = teacher_embeds.new_zeros(teacher_embeds.shape[2], dtype=torch.bool)
                    ans_mask = teacher_embeds.new_zeros(teacher_embeds.shape[2], dtype=torch.bool)
                    if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0: # no image in conversation
                        pass
                    else:
                        num_img_token = 576
                        image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].item()
                        img_mask[image_token_indices:image_token_indices+num_img_token] = True
                        ans_mask = stu_labels[batch_idx] != IGNORE_INDEX
                        ans_mask = ans_mask & stu_attention_mask[batch_idx] # remove padding
                    image_masks.append(img_mask)
                    answer_masks.append(ans_mask)

                image_masks = torch.stack(image_masks, dim=0)
                answer_masks = torch.stack(answer_masks, dim=0)
                
                masks = (image_masks.unsqueeze(2) * answer_masks.unsqueeze(1)).unsqueeze(1)
                masks = (answer_masks.unsqueeze(2) * answer_masks.unsqueeze(1)).unsqueeze(1)
                masks = (answer_masks.unsqueeze(2) * image_masks.unsqueeze(1)).unsqueeze(1)

                attn_loss = (attn_loss * masks).sum() / (masks.sum() + 1e-6)

                loss = loss + attn_loss * 5.0

        if self.args.align_vision_tower:
            image_masks, answer_masks = self.get_image_masks(input_ids, stu_labels, stu_attention_mask)
            
            masks = torch.logical_or(image_masks, answer_masks)

            mse_loss = F.mse_loss(stu_inputs_embeds, teacher_inputs_embeds, reduction='none')

            # TODO: check mse_loss[image_masks].sum() >> mse_loss[~image_masks].sum()
            mse_loss = (mse_loss * masks.unsqueeze(-1)).sum() / (masks.sum() + 1e-6)
            loss = loss + mse_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=student_result.logits,
            past_key_values=student_result.past_key_values,
            hidden_states=student_result.hidden_states,
            attentions=student_result.attentions,
        )

    def get_image_masks(self, input_ids, labels, attention_mask):
        image_masks, answer_masks = [], []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            img_mask = labels.new_zeros(labels.shape[1], dtype=torch.bool)
            ans_mask = labels.new_zeros(labels.shape[1], dtype=torch.bool)
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0: # no image in conversation
                pass
            else:
                num_img_token = labels.shape[1] - input_ids.shape[1] + 1 # 196 or 576
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].item()
                img_mask[image_token_indices:image_token_indices+num_img_token] = True
            ans_mask = labels[batch_idx] != IGNORE_INDEX
            x = ans_mask.sum()
            ans_mask = ans_mask & attention_mask[batch_idx] # remove padding
            assert x == ans_mask.sum()
            image_masks.append(img_mask)
            answer_masks.append(ans_mask)
        image_masks = torch.stack(image_masks, dim=0)
        answer_masks = torch.stack(answer_masks, dim=0)
        return image_masks, answer_masks