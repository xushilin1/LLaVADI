* output/distill/finetune/llava_MobileLLaMA-2。7B-Chat_align_image_tokens_align_logits
    * textvqa：49.68

* output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_align_logits
    * textvqa：49.5

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_align_logits_exp3
    * bs = 2 * 8 * 8 
    * epoch = 2
    * KL散度，只有answer token产生loss
    * textvqa: 50.3

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp20
    * bs = 2 * 8 * 8 
    * epoch = 2
    * KL散度，只有answer token产生loss, 蒸馏loss * 5

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp8
    * bs = 2 * 8 * 8
    * epoch = 1
    * KL散度，loss的权重调整为 5
    * textvqa: 49.8

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_align_logits_exp4
    * bs = 2 * 8 * 8
    * align_logits: 所有valid token都产生loss
    * textvqa: 46.7

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp5
    * bs = 2 * 8 * 8
    * align affinity（各自取最后20层）
    * textvqa：46.3

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp6
    * align affinity（最后20层）+ align_logits（最后一层）
    * textvqa: 48.9
    
* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp7
    * align affinity；均匀采用10层
    * textvqa: 48.0
    * 相比于exp6，均匀采用10层效果更好。

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp9 
    * 使用20层的LLaVA，其他保持一致，使用LLaVA-13b的前20层初始化
    * align_logits， loss*5
    * textvqa：54.1

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp10
    * align_logits， loss*5 
    * 使用20层的LLaVA，其他保持一致，使用LLaVA-13b的隔层初始化
    * textvqa: 45.8
    
    
* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp11
    * 均匀选取10个hidden embeddings进行蒸馏
    * 共享的两层Linear
    * textvqa: 45.1
    
* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp12
    * align_logits；去掉student原本的loss；只保留蒸馏loss；蒸馏loss * 5,
    * 采用0.7的temperature
    * textvqa：49.9

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp18
    * align_logits；去掉student原本的loss；只保留蒸馏loss；蒸馏loss * 5,
    * 采用1.0的temperature
    * textvqa: 49.9

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp14
    * align_logits；去掉student原本的loss；只保留蒸馏loss；蒸馏loss * 5,
    * 采用0.1的temperature
    * textvqa：46.3

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp13
    * align_logits；去掉student原本的loss；只保留蒸馏loss；蒸馏loss * 5,
    * 采用0.25的temperature
    * textvqa：48.7
    * 和15相比，只使用蒸馏loss只掉了0.1；和12，14相比，温度系数影响更大；

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp15
    * align_logits, 原本的loss+蒸馏loss, 蒸馏loss * 5,
    * 采用0.25的temperature
    * textvqa: 48.8

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp16
    * llava 20层，使用隔层初始化
    * 蒸馏hidden state, 均匀使用10层，由于维度一致，不使用Linear
    * 不使用Flash Attention

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp17
    * 4 * 8 * 2；lr=1e-5
    * llava 20层，使用隔层初始化
    * 蒸馏hidden state和attention map, attention map使用最后一层（OOM），由于维度一致，不使用Linear
    * 不使用Flash Attention
    * textvqa: 46.3

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp19
    * 2 * 8 * 8；(align_logits,温度0.7 + align_hidden_embeds,均匀采10层，共享的两层Linear) * 5
    * align_logits的temperature设置成0.7
    * textvqa: 46.6

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp21
    * align_hidden_embeds；最后一层使用mse * 5
    * student维度升到teacher维度
    * textvqa: 48.7

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp22
    * align_hidden_embeds；最后一层使用mse * 5；align_logits * 5
    * student维度升到teacher维度
    * textvqa: 50.7

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp36
    * hidden_embeds（仅使用answer token）；最后一层使用mse * 5；align_logits * 5
    * student维度升到teacher维度
    * textvqa: 49.4

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp37
    * aligh_affinity; hidden_embeds（仅使用answer token）；align_logits * 5
    * student维度升到teacher维度
    * textvqa: 49.6

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp42
    * aligh_affinity; hidden_embeds（使用所有token）align_logits * 5
    * student维度升到teacher维度
    * textvqa: 

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp23
    * 使用20层的LLaVA，使用LLaVA-13b的最后20层初始化（貌似没有初始化成功）
    * align_logits， loss*5

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp31
    * 使用20层的LLaVA，使用LLaVA-13b的前面20层初始化； 使用LLaVA-13B的projector
    * align_logits * 5
    * textvqa: 54.3

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp38
    * 使用20层的LLaVA，使用LLaVA-13b的前面20层初始化； 使用LLaVA-13B的projector
    * align_logits * 5, hidden_embeds（仅使用answer token）； 
    * textvqa: 54.1

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp39
    * 使用20层的LLaVA，使用LLaVA-13b的前面20层初始化； 使用LLaVA-13B的projector
    * align_logits * 5, hidden_embeds（仅使用answer token）；attention_map 
    * align_attn_map(image token to answer token)
    * textvqa: 54.0

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp40
    * 使用20层的LLaVA，使用LLaVA-13b的前面20层初始化； 使用LLaVA-13B的projector
    * align_logits * 5, hidden_embeds（仅使用answer token）；attention_map 
    * align_attn_map(answer token to answer token)
    * textvqa: 54.0

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp41
    * 使用20层的LLaVA，使用LLaVA-13b的前面20层初始化； 使用LLaVA-13B的projector
    * align_logits * 5, hidden_embeds（仅使用answer token）；attention_map 
    * align_attn_map(answer token to image token)
    * textvqa: 54.4


* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp32
    * 使用10层的LLaVA，使用LLaVA-13b的前面10层初始化； 使用LLaVA-13B的projector
    * align_logits * 5
    * textvqa: 30.5

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp34
    * 使用10层的LLaVA，使用LLaVA-13b的前面10层初始化； 使用LLaVA-13B的projector
    * align_logits * 5, align_hidden_states * 5 （最后一层）
    * textvqa: 30.2

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp35
    * 使用10层的LLaVA，使用LLaVA-13b的前面10层初始化； 使用LLaVA-13B的projector
    * align_logits * 5, align_hidden_states * 5 （最后一层）
    * epoch=3

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp43
    * 使用20层的LLaVA，使用LLaVA-13b的前面20层初始化；
    * openai/clip-vit-base；位置编码随机初始化；维度768和teacher的维度不一致；随机初始化映射层
    * aligh_affinity; hidden_embeds（使用所有token）align_logits * 5
    * align_vision_tower（teacher使用LLaVA，经过映射层后的特征维度可以保持一致）
    * epoch=6

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp44
    * 使用20层的LLaVA，使用LLaVA-13b的前面20层初始化；
    * openai/clip-vit-base；位置编码随机初始化；维度768和teacher的维度不一致；随机初始化映射层
    * aligh_affinity; hidden_embeds（使用所有token）align_logits * 5
    * align_vision_tower（teacher使用LLaVA，经过映射层后的特征维度可以保持一致）
    * epoch=1


* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp45
    * 使用20层的LLaVA，使用LLaVA-13b的前面20层初始化；
    * 随机初始化映射层和patch—embedding；不使用align_vision_tower的loss
    * aligh_affinity; hidden_embeds（使用所有token）align_logits * 5
    * align_vision_tower（teacher使用LLaVA，经过映射层后的特征维度可以保持一致）
    * epoch=1


* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp46
    * 使用30层的LLaVA，使用LLaVA-13b的前面30层初始化；
    * aligh_affinity; hidden_embeds（使用所有token）align_logits * 5; align_attn_map;
    * epoch=1
    * textvqa: 57.2

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp47
    * 使用30层的LLaVA，使用LLaVA-13b的前面30层初始化；
    * aligh_affinity; hidden_embeds（使用所有token）align_logits * 5; align_attn_map;
    * epoch=3

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp48
    * 使用30层的LLaVA，使用LLaVA-13b的前面30层初始化；使用llava的映射层
    * align_logits * 5;
    * epoch=1
    * textvqa: 57.4

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp49
    * 使用30层的LLaVA，使用LLaVA-13b的前面30层初始化；使用llava的映射层
    * epoch=1
    
    
* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp50
    * 使用30层的LLaVA，使用LLaVA-13b的前面30层初始化；使用llava的映射层
    * align_logits; 不使用原本自回归loss
    * epoch=1
    * textvqa: 56.5

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp51
    * 使用20层的LLaVA，使用LLaVA-13b-30-layer的前面20层和映射层初始化
    * align_logits
    * epoch=1
    * textvqa: 50.1

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp53
    * student: clip-vit-base-patch16 + MobileLLaMA-2.7B-Chat
    * teacher: LLaVA-13B
    * align_logits(仅有answer token产生loss); 使用5120个token

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp54
    * student: clip-vit-base-patch16 + MobileLLaMA-2.7B-Chat
    * teacher: LLaVA-13B
    * align_logits(仅有answer token产生loss)；使用2048个token（默认值）
    * textvqa：41.1

* ./output/distill/finetune/llava_MobileLLaMA-2.7B-Chat_exp55
    * student: clip-vit-base-patch16 + MobileLLaMA-2.7B-Chat
    * teacher: LLaVA-13B
    * align_logits(仅有answer token产生loss); 使用2048个token（默认值）；
    * finetune最后6层vit