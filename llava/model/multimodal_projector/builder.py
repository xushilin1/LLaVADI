import torch
import torch.nn as nn
import re
import math
from functools import partial
from timm.layers.norm_act import LayerNormAct2d
from torchvision.ops.misc import SqueezeExcitation as SElayer
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class LDPBlock(nn.Module):
    # Lightweight Downsample Projector Block

    def __init__(self, config=None):
        super().__init__()

        inc, ouc = config.mm_hidden_size, config.hidden_size
        layer_norm = partial(LayerNormAct2d, act_layer=None)
        se_layer = partial(SElayer, scale_activation=nn.Hardsigmoid)
        self.mlp = nn.Sequential(
            nn.Identity(), nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc)
        )
        self.mb_block = nn.Sequential(
            nn.Identity(),
            InvertedResidual(InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 1, 1, 1), layer_norm, se_layer),
            InvertedResidual(InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 2, 1, 1), layer_norm, se_layer)
        )

    def forward(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        x = self.mlp(x) 
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.mb_block(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x

class FeatureIRLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class TokenDownLayer(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.dwn = nn.Sequential(
            nn.AdaptiveAvgPool2d(shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.dwn(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
class PosInjectLayer(nn.Module):
    # https://github.com/Meituan-AutoML/Twins/blob/main/gvt.py
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.peg = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=True, groups=out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        cnn_feat = x.transpose(1, 2).view(b, c, h, h)
        x = self.peg(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        return x

class LDPNetProjector(nn.Module):
    
    def __init__(self, config=None):
        super().__init__()
        self.model = LDPBlock(config)

    def forward(self, x):
        return self.model(x)

class LDPNetV2Projector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.mlp = FeatureIRLayer(inc, ouc)
        self.dwn = TokenDownLayer((12, 12))
        self.peg = PosInjectLayer(ouc, ouc, stride=1)

    def forward(self, x):
        x = self.mlp(x)
        x = self.dwn(x)
        x = self.peg(x)
        return x


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()
    
    if projector_type == 'ldpnetv2':
        return LDPNetV2Projector(config)
    
    if projector_type == 'ldpnet':
        return LDPNetProjector(config)
    raise ValueError(f'Unknown projector type: {projector_type}')
