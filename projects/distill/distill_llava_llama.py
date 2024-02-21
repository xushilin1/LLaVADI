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
            for i in range(labels.shape[0]):
                if self.args.align_logits_all:
                    teacher_logits = teacher_result.logits[i][stu_attention_mask[i]]
                    stu_logits = student_result.logits[i][stu_attention_mask[i]]
                else:
                    stu_shift_logits = student_result.logits[i, :-1, :].contiguous()
                    stu_shift_labels = stu_labels[i, 1:].contiguous()
                    stu_logits = stu_shift_logits[stu_shift_labels != IGNORE_INDEX]
                    
                    teacher_shift_logits = teacher_result.logits[i, :-1, :].contiguous()
                    teacher_shift_labels = teacher_labels[i, 1:].contiguous()
                    teacher_logits = teacher_shift_logits[teacher_shift_labels != IGNORE_INDEX]
                
                if self.args.mse_distill:
                    distill_loss += F.mse_loss(stu_logits, teacher_logits)
                else:
                    distill_loss += F.kl_div(
                        F.log_softmax(stu_logits / 0.7, dim=-1),
                        F.softmax(teacher_logits / 0.7, dim=-1),
                        reduction='batchmean',
                    ) * 0.7 * 0.7
            distill_loss /= labels.shape[0]
            distill_loss *= 5.0
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

        if self.args.align_hidden_embeds:
            teacher_embeds = torch.stack(teacher_result.hidden_states, dim=1) #(bs, layers, N, 5120)
            student_embeds = torch.stack(student_result.hidden_states, dim=1) #(bs, layers, N, 2048)
            
            # teacher_layer = torch.linspace(0, teacher_embeds.shape[1]-1, steps=10).long()
            # student_layer = torch.linspace(0, student_embeds.shape[1]-1, steps=10).long()
            teacher_layer = [-1]
            student_layer = [-1]
            teacher_embeds = teacher_embeds[:, teacher_layer, :, :] # (bs, 1, N, 5120)
            student_embeds = student_embeds[:, student_layer, :, :]

            # teacher_embeds = self.embed_projector(teacher_embeds)
            student_embeds = self.embed_projector(student_embeds)
            image_masks, answer_masks = [], []
            for batch_idx, cur_input_ids in enumerate(input_ids):
                ans_mask = stu_labels[batch_idx] != IGNORE_INDEX
                answer_masks.append(ans_mask)
            
            mse_loss = F.mse_loss(student_embeds, teacher_embeds, reduction='none')
            answer_masks = torch.stack(answer_masks, dim=0).unsqueeze(1).unsqueeze(-1)
            answer_masks = answer_masks.expand_as(mse_loss)
            answer_masks = torch.ones_like(answer_masks)
            answer_masks = answer_masks * stu_attention_mask.unsqueeze(1).unsqueeze(-1)
            mse_loss = (mse_loss * answer_masks).sum() / (answer_masks.sum() + 1e-6)

            loss = loss + mse_loss * 5.0

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
                ans_mask = ans_mask & attention_mask[batch_idx] # remove padding
            image_masks.append(img_mask)
            answer_masks.append(ans_mask)
        image_masks = torch.stack(image_masks, dim=0)
        answer_masks = torch.stack(answer_masks, dim=0)
        return image_masks, answer_masks