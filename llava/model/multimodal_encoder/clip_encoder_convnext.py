import torch
import torch.nn as nn
import open_clip
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

class CLIPVisionTowerConvNext(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        # self.select_layer = args.mm_vision_select_layer
        # self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            raise NotImplementedError

    def load_model(self):
        self.img_size = 512
        self.image_processor = CLIPImageProcessor(do_resize=True, size={"shortest_edge":self.img_size}, resample=3,  do_center_crop=True, crop_size={"height": self.img_size, "width": self.img_size},
                                                do_rescale=True, rescale_factor=0.00392156862745098, do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073],
                                                image_std=[0.26862954, 0.26130258, 0.27577711], do_convert_rgb=True, )
        model_name, pretrained = self.vision_tower_name.split('/')
        model_cfg = open_clip.get_model_config(model_name)
        pretrained_cfg = open_clip.get_pretrained_cfg(model_name, pretrained)
        checkpoint_path = open_clip.download_pretrained(pretrained_cfg, cache_dir=None)
        model_cfg.pop('text_cfg')
        self.vision_tower = open_clip.model._build_vision_tower(**model_cfg)
        # open_clip.load_checkpoint(visual, checkpoint_path)
        self.vision_tower.load_state_dict(torch.load(checkpoint_path), strict=False)

        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

        self.hidden_size = model_cfg['embed_dim']

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            image_features_dict = []
            for image in images:
                image_feature_dict = self.extract_features(image.unsqueeze(0))
                image_features_dict.append(image_feature_dict)
                image_feature = image_feature_dict['res4']
                image_feature = image_feature.reshape(*image_feature.shape[:2],-1).permute(0,2,1)
                image_features.append(image_feature)
        else:
            image_features_dict = self.extract_features(images)
            image_features = image_features_dict['res4']
            image_features = image_features.reshape(*image_features.shape[:2],-1).permute(0,2,1)
        
        return image_features
        # return image_features, image_features_dict
    
    def extract_features(self, x):
        self.eval()
        with torch.no_grad():
            out = {}
            x = x.to(self.vision_tower.trunk.stem.state_dict()['1.bias'].dtype)
            x = self.vision_tower.trunk.stem(x)
            out['stem'] = x.contiguous() 
            for i in range(4):
                x = self.vision_tower.trunk.stages[i](x)
                out[f'res{i+2}'] = x.contiguous() 
            
            x = self.vision_tower.trunk.norm_pre(x)
            out['clip_vis_dense'] = x.contiguous()
            return out

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device
