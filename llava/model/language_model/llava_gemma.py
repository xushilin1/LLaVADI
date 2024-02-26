from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


from transformers import AutoConfig, AutoModelForCausalLM

from .gemma.modeling_gemma import GemmaConfig, GemmaModel, GemmaForCausalLM
from transformers.generation.utils import GenerateOutput

from transformers.modeling_outputs import CausalLMOutputWithPast


from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from ..model_factory import register_model, register_tokenizer

class LlavaGemmaConfig(GemmaConfig):
    model_type = "llava_gemma"


class LlavaGemmaModel(LlavaMetaModel, GemmaModel):
    config_class = LlavaGemmaConfig

    def __init__(self, config: LlavaGemmaConfig):
        super(LlavaGemmaModel, self).__init__(config)
        self.gradient_checkpointing = False

@register_model('gemma')
class LlavaGemmaForCausalLM(GemmaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaGemmaConfig

    def __init__(self, config):
        super(GemmaForCausalLM, self).__init__(config)
        self.model = LlavaGemmaModel(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

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
        # image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                # image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        # image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                # image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlavaGemmaModel):
            module.gradient_checkpointing = value

@register_tokenizer('gemma')
def get_tokenizer():
    from .gemma.tokenization_gemma import GemmaTokenizer
    def post_init(tokenizer):
        return tokenizer
    return GemmaTokenizer, post_init

AutoConfig.register("llava_gemma", LlavaGemmaConfig)
AutoModelForCausalLM.register(LlavaGemmaConfig, LlavaGemmaForCausalLM)