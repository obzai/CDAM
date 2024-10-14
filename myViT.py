import torch
from cdam.dino.vision_transformer import VisionTransformer
import math
from torch import nn

class MyVisionTransformer(VisionTransformer):
    """Vision Transformer with a linear head for classification. Custom implementation to remove token embeddings during inference."""
    def forward(self, x, remove=None):
        x = self.prepare_tokens(x, remove)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)
        return x[:, 0]
    
    def prepare_tokens(self, x, remove=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        # pos_encoding = self.interpolate_pos_encoding(x, w, h)
        pos_encoding = super().interpolate_pos_encoding(x,w,h)
        x = x + pos_encoding
        # remove patch embeddings defined in "remove"
        if remove != None:
            
            num_patches = x.shape[1] 
            t = torch.arange(num_patches)
            idx_to_keep = [i for j, i in enumerate(t) if j not in remove]
            x = x[:, idx_to_keep, :]
        x = self.pos_drop(x)
        return x

    # def interpolate_pos_encoding(self, x, w, h):
    #     npatch = x.shape[1] - 1
    #     N = self.pos_embed.shape[1] - 1
    #     if npatch == N and w == h:
    #         return self.pos_embed
    #     class_pos_embed = self.pos_embed[:, 0]
    #     patch_pos_embed = self.pos_embed[:, 1:]
    #     dim = x.shape[-1]
    #     w0 = w // self.patch_embed.patch_size
    #     h0 = h // self.patch_embed.patch_size
    #     # we add a small number to avoid floating point error in the interpolation
    #     # see discussion at https://github.com/facebookresearch/dino/issues/8
    #     w0, h0 = w0 + 0.1, h0 + 0.1
    #     patch_pos_embed = nn.functional.interpolate(
    #         patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
    #         scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
    #         mode='bicubic',
    #     )
    #     assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    #     patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    #     return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)