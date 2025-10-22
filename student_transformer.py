"""
student_transformer.py LAYER NO.1
----------------------
This file implements a Vision Transformer (ViT) architecture in PyTorch. It includes the ViT class, 
which splits images into patches, embeds them, and processes them with a multi-layer transformer. 
The code also defines supporting modules such as Residual, PreNorm, FeedForward, and Attention. 
This transformer is used as an encoder in larger models for image understanding tasks.

Why: This file exists to provide a modular Vision Transformer encoder, which can be used as a 
backbone for various vision tasks, leveraging the power of self-attention for image understanding.
"""
# -*- coding: utf-8 -*-
"""
@author: Pankaj Mishra
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from peft import apply_lora
from peft import apply_lora

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            # nn.ReLU(inplace=True)
           
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
           
        )

    def forward(self, x, mask=None):
        # x: input embeddings, shape [batch, tokens, embed_dim]

        b, n, _, h = *x.shape, self.heads
        # b = batch size, n = number of tokens, h = number of heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # Linear projection of x to produce Q, K, V concatenated, then split into 3 tensors

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # Reshape each tensor to [batch, heads, tokens, head_dim] so each head can attend separately

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # Compute attention scores: Q·K^T / sqrt(d), shape [batch, heads, tokens, tokens]

        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        # Apply mask (if provided) to prevent attention to certain positions, e.g., padding or future tokens

        attn = dots.softmax(dim=-1)
        # Softmax over last dimension → convert scores to attention weights (sum = 1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # Multiply attention weights by V → context-aware embeddings for each head

        out = rearrange(out, 'b h n d -> b n (h d)')
        # Concatenate heads back together to restore original embedding size

        out = self.to_out(out)
        # Final linear projection to combine multi-head outputs

        return out
        # Return updated token embeddings with context from other tokens


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        # Apply LoRA to all Linear layers in transformer if desired
        self.transformer = apply_lora(self.transformer, r=4, alpha=1.0)

        self.to_cls_token = nn.Identity()

        

    def forward(self, img, mask = None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p) # splits the image into patches of patch size p which is 64 from VT_AE
        x = self.patch_to_embedding(x) # linear projection of each patch into dim (512 from VT_AE)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)] # applies positional embedding. it just adds position embedding to x withc is the patch embeddings
       

        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:,1:,:])
       
        return x

if __name__ =="__main__":

    v = ViT(
        image_size = 512,
        patch_size = 32,
        num_classes = 1,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 1024,
       
    )
    
    img = torch.randn(1, 3, 512, 512)
    mask = torch.ones(1, 16, 16).bool() # optional mask, designating which patch to attend to
    print(v)
    preds = v(img, mask = mask) # (1, 1000)
    print(preds)