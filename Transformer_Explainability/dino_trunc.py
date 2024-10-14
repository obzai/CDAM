from dino.vision_transformer import VisionTransformer
import torch
from torch import nn
from functools import partial

# This truncated version returns the CLS after the final attention layer, so before it goes through the final MLP
class ViTTruncated(VisionTransformer):
    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks[:-1]:
            x = blk(x)
        x = self.blocks[-1](x, return_no_mlp=True)
        return x[:, 0]
        
def dino_trunc():
    model_trunc = ViTTruncated(
        patch_size=8,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    state_dict = torch.hub.load_state_dict_from_url(
        url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
        map_location="cpu",
    )
    model_trunc.load_state_dict(state_dict, strict=True)
    return model_trunc