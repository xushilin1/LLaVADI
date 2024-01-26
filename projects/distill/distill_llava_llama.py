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
                 student_model = None,
                 student_tokenizer = None,
                 teacher_model = None,
                 teacher_tokenizer = None,):
        super(DistillModel, self).__init__()
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

        student_result = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            images=images,
            return_dict=return_dict
        )
        torch.cuda.empty_cache()
        teacher_result = self.teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            images=images,
            return_dict=return_dict
        )
        distill_loss = 0
        if True:
            valid_num = (labels != -100).sum(-1)
            student_logits = student_result.logits[:, :-1, :].contiguous()
            teacher_logits = teacher_result.logits[:, :-1, :].contiguous()

            for i in range(valid_num.shape[0]):
                student_logit = student_logits[i, -valid_num[i]:, :]
                teacher_logit = teacher_logits[i, -valid_num[i]:, :]
                distill_loss += F.kl_div(
                    F.log_softmax(student_logit / 0.7, dim=-1),
                    F.softmax(teacher_logit / 0.7, dim=-1),
                    reduction='batchmean',
                ) * 0.7 * 0.7
            distill_loss /= valid_num.shape[0]
        
        return CausalLMOutputWithPast(
            loss=student_result.loss + distill_loss,
            logits=student_result.logits,
            past_key_values=student_result.past_key_values,
            hidden_states=student_result.hidden_states,
            attentions=student_result.attentions,
        )
