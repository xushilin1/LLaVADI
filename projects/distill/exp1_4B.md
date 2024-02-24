* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp1
    * LLaVA-13B蒸馏；align_logits * 5
    * textvqa:45.1

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp10
    * align_logits * 5; answer only; 
    * epoch=3

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp6
    * LLaVA-13B蒸馏；align_logits * 1
    * textvqa: 44.43

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp7
    * align_hidden_embeds * 1; answer only; 
    * textvqa: 43.98

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp8
    * align_hidden_embeds(**cosine similarity**) * 1; answer only; 
    * textvqa: 43.76

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp9
    * align_affinity * 1;
    * 

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp2
    * LLaVA-13B蒸馏；align_logits（CE）* 5
    * textvqa: 44.7

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp3
    * LLaVA-13B蒸馏；align_logits * 5 + align_hidden_embeds * 5(answer token only)
    * textvqa:44.75

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp4
    * LLaVA-13B蒸馏；align_logits * 5 + align_hidden_embeds(**cosine similarity**) * 5 (answer token only)
    * textvqa: 45.1

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp5
    * LLaVA-13B蒸馏；align_logits(CE) + align_hidden_embeds(**cosine similarity**) (answer token only)
    * textvqa: 45.24


* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp11
    * align_logits * 5; answer only; 
    * 两阶段蒸馏（pretrain+finetune）