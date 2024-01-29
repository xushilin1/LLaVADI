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
        images: Optional[torch.FloatTensor] = None,
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
            images
        )

        teacher_result = self.teacher_model.forward(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
            position_ids=teacher_position_ids,
            past_key_values=teacher_past_key_values,
            inputs_embeds=teacher_inputs_embeds,
            labels=teacher_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            images=images,
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
            images
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
                    
                    # FIXME: why stu_labels and teacher_labels are different ???
                    # answer_masks = (stu_labels[batch_idx] != IGNORE_INDEX)[0]
                    answer_masks = (teacher_labels[batch_idx] != IGNORE_INDEX)
                    num_ans_token = answer_masks.sum()

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
            output_attentions=output_attentions,
            output_hidden_states=True,
            images=images,
            return_dict=return_dict
        )
        distill_loss = 0
        if self.args.align_logits:
            for i in range(labels.shape[0]):
                stu_shift_logits = student_result.logits[i, :-1, :].contiguous()
                stu_shift_labels = stu_labels[i, 1:].contiguous()
                stu_logits = stu_shift_logits[stu_shift_labels != IGNORE_INDEX]
                
                teacher_shift_logits = teacher_result.logits[i, :-1, :].contiguous()
                teacher_shift_labels = teacher_labels[i, 1:].contiguous()
                teacher_logits = teacher_shift_logits[teacher_shift_labels != IGNORE_INDEX]
                
                distill_loss += F.kl_div(
                    F.log_softmax(stu_logits / 0.7, dim=-1),
                    F.softmax(teacher_logits / 0.7, dim=-1),
                    reduction='batchmean',
                ) * 0.7 * 0.7
            distill_loss /= labels.shape[0]
        
        return CausalLMOutputWithPast(
            loss=student_result.loss + distill_loss,
            logits=student_result.logits,
            past_key_values=student_result.past_key_values,
            hidden_states=student_result.hidden_states,
            attentions=student_result.attentions,
        )
