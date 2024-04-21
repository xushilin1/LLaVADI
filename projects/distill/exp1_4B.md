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
    * align_logits * 5; answer only; pretrain
    * 两阶段蒸馏（pretrain+finetune）

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp14
    * align_logits * 5; answer only;  finetune
    * 两阶段蒸馏（pretrain+finetune）

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp15
    * align_logits * 5; answer only;  finetune；继续放开vit
    * 两阶段蒸馏（pretrain+finetune）


* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp12
    * 蒸馏ViT-Base；align_logits（answer token only）
    * textvqa: 37.9

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp23
    * 蒸馏ViT-Base；align_logits（answer token only）

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp13
    * 2.7B MobileLLaMA
    * 蒸馏ViT-Base；align_logits（answer token only）
    * textvqa: 42.8


./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp20
    * 1.4B蒸馏；蒸馏ViT-Base; 使用baseline进行初始化


./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp24
    * 1.4B蒸馏；align_logits, align_hidden_embeds; 
    * 8 * 16

./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp25
    * 1.4B蒸馏；align_logits, align_hidden_embeds;
    * 8 * 32（pretrain）

./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp26
    * 1.4B蒸馏；align_logits, align_hidden_embeds(all tokens);
    * 8 * 32（pretrain）

./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp27
    * 1.4B蒸馏；align_logits; 25
    * （finetune）
    * textvqa:41.9; sqa:35.7; pope: 84.2; mme: 998.4

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp28
    * 1.4B蒸馏；align_logits; tune vit；
    * （finetune）
    * textvqa: 41.9; sqa: 35.8; mme:1008.5 pope:84.1

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp31
    * 1.4B蒸馏；align_logits; align_affinity；tune vit；
    * finetune）
    * textvqa: 41.8; sqa: 37.1; mme: 992; pope: 84.0

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp29
    * 1.4B蒸馏；align_logits；offline data
    * textvqa：45.6

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp30
    * 2.4B蒸馏；align_logits；merged data
    * textvqa: 49.6; sqa: 52.2

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp32
    * 不要蒸馏；merged data
    * textvqa: 43.2

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp33
    * 不要蒸馏；2.7B；offline data
    * textvqa: 47.7

* ~~./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp34~~
    * 2.7B; finetune之后使用offline data继续训练；

* ~~./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp35~~
    * 1.4B; finetune之后使用offline data继续训练；

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp36 
    * align_contrastive_affinity； align_logits

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp37
    * align_contrastive_affinity； align_logits；更小的loss

* ./output/distill/finetune/llava_MobileLLaMA-1.4B-Chat_exp38
    * align_sparse_logits
