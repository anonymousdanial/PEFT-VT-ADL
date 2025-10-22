"""

FIRST TRANSFORMER LAYER



VT_AE.py 
---------
This file defines the VT_AE class, a Vision Transformer-based AutoEncoder architecture in PyTorch. 
It combines a Vision Transformer (ViT) encoder, a capsule-based bottleneck (DigitCaps), 
and a convolutional decoder. The model is designed for image encoding and reconstruction, 
with optional noise injection for regularization. It is used for unsupervised or self-supervised 
learning tasks on images.

Why: This file exists to implement a transformer-based autoencoder architecture for unsupervised learning, 
enabling the extraction of rich, high-level representations from images for downstream tasks or anomaly 
detection.
"""
# -*- coding: utf-8 -*-
"""
@author: Pankaj Mishra
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from student_transformer import ViT
import model_res18 as M
from einops import rearrange
import spatial as S
from transformers import ViTModel, ViTConfig


#### NETWORK DECLARATION ####
# torch.autograd.set_detect_anomaly(True) # this is to check any problem in the network by backtracking

class VT_AE(nn.Module):
    def __init__(self, image_size = 512,
                    patch_size = 64,
                    num_classes = 1,
                    dim = 512,
                    depth = 6,
                    heads = 8,
                    mlp_dim = 1024,
                    train= True,
                    pretrained_vit_name: str = None,
                    load_pretrained: bool = False):

        super(VT_AE, self).__init__()
        # Replace the local transformer with a (optionally) pretrained HuggingFace ViT encoder.
        # A small wrapper is used to keep the original forward contract: vt(x, mask) -> (batch, num_patches, dim)
        self.vt = PretrainedViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            target_dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pretrained_name=pretrained_vit_name if load_pretrained else None
        )
        
     
        self.decoder = M.decoder2(8)
        # self.G_estimate= mdn1.MDN() # Trained in modular fashion
        self.Digcap = S.DigitCaps(in_num_caps=((image_size//patch_size)**2)*8*8, in_dim_caps=8)
        self.mask = torch.ones(1, image_size//patch_size, image_size//patch_size).bool()#.cuda()
        self.Train = train
        
        if self.Train:
            print("\nInitializing network weights.........")
            initialize_weights(self.vt, self.decoder)

    def forward(self,x):
        b = x.size(0)
        encoded = self.vt(x, self.mask)
        if self.Train:
            encoded = add_noise(encoded)
        encoded1, vectors = self.Digcap(encoded.view(b,encoded.size(1)*8*8,-1)) # apparently this is where the "reconstruction matrix" is generated 
        recons = self.decoder(encoded1.view(b,-1,8,8)) # this is where the image is reconstructed from the "reconstruction matrix"
        # pi, mu, sigma = self.G_estimate(encoded)       
        # return encoded, pi, sigma, mu, recons
            
        return encoded, recons
    
# Initialize weight function
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class PretrainedViTEncoder(nn.Module):
    """Wrapper that exposes an encoder-only ViT interface compatible with the original `ViT` used
    in this project. It can load a HuggingFace pretrained ViTModel (encoder-only) and will
    optionally resize positional embeddings and project the model's hidden size to `target_dim`.

    The forward signature matches the original: forward(img, mask=None) -> (batch, num_patches, target_dim)
    """
    def __init__(self, image_size: int, patch_size: int, target_dim: int, depth: int, heads: int, mlp_dim: int, pretrained_name: str = None):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        if pretrained_name is not None:
            # load pretrained ViT encoder
            self.model = ViTModel.from_pretrained(pretrained_name)
            config = self.model.config
        else:
            # build a ViT config that mirrors the original student transformer parameters
            config = ViTConfig(
                image_size=image_size,
                patch_size=patch_size,
                hidden_size=target_dim,
                num_hidden_layers=depth,
                num_attention_heads=heads,
                intermediate_size=mlp_dim,
            )
            self.model = ViTModel(config)

        self.emb_dim = self.model.config.hidden_size

        # If the pretrained model was trained with a different image/patch grid size, resize its pos embeddings.
        if (hasattr(self.model.config, 'image_size') and self.model.config.image_size is not None):
            trained_image_size = self.model.config.image_size
            if trained_image_size != image_size:
                try:
                    self._resize_pos_emb(trained_image_size)
                except Exception:
                    # If resizing fails, keep the original pos embeddings. Downstream may still work but
                    # could lead to a mismatch in attention behavior.
                    pass

        # Project embedding dimension to target_dim if needed
        if self.emb_dim != target_dim:
            self.proj = nn.Linear(self.emb_dim, target_dim)
        else:
            self.proj = nn.Identity()

    def _resize_pos_emb(self, trained_image_size: int):
        """Interpolate the pretrained positional embeddings to match the new image/patch grid."""
        pos_emb = self.model.embeddings.position_embeddings  # (1, seq_len, dim)
        seq_len = pos_emb.shape[1]
        num_patches_old = seq_len - 1
        old_grid = int(math.sqrt(num_patches_old))
        num_patches_new = (self.image_size // self.patch_size) ** 2
        new_grid = int(math.sqrt(num_patches_new))
        if old_grid == new_grid:
            return

        cls_token = pos_emb[:, :1, :]
        patch_pos = pos_emb[:, 1:, :]
        # reshape to (1, dim, H, W) for interpolation
        patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(new_grid, new_grid), mode='bilinear', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
        new_pos = torch.cat([cls_token, patch_pos], dim=1)
        # replace the model's positional embeddings
        self.model.embeddings.position_embeddings = nn.Parameter(new_pos)
        # Ensure the model's config matches the new image/patch grid so
        # the internal embedding computations use the updated sizes.
        # This prevents shape mismatches (for example: pretrained 384-> new 512)
        try:
            self.model.config.image_size = self.image_size
            # patch_size is sometimes an attribute in the HF config
            self.model.config.patch_size = self.patch_size
        except Exception:
            # If config cannot be updated for any reason, we still keep the new pos emb
            pass

    def forward(self, img, mask=None):
        # The HF ViTModel expects `pixel_values` shaped [batch, channels, height, width]
        outputs = self.model(pixel_values=img)
        x = outputs.last_hidden_state  # (b, seq_len, emb_dim)
        # drop cls token to match previous student implementation
        x = x[:, 1:, :]
        x = self.proj(x)
        return x
                
##### Adding Noise ############

def add_noise(latent, noise_type="gaussian", sd=0.2):
    """Here we add noise to the latent features concatenated from the 4 autoencoders.
    Arguements:
    'gaussian' (string): Gaussian-distributed additive noise.
    'speckle' (string) : Multiplicative noise using out = image + n*image, where n is uniform noise with specified mean & variance.
    'sd' (integer) : standard deviation used for geenrating noise

    Input :
        latent : numpy array or cuda tensor.

    Output:
        Array: Noise added input, can be np array or cuda tnesor.
    """
    assert sd >= 0.0
    if noise_type == "gaussian":
        mean = 0.

        n = torch.distributions.Normal(torch.tensor([mean]), torch.tensor([sd]))
        noise = n.sample(latent.size()).squeeze(-1)#.cuda()
        latent = latent + noise
        return latent

    if noise_type == "speckle":
        noise = torch.randn(latent.size())#.cuda()
        latent = latent + latent * noise
        return latent

if __name__ == "__main__":
    from torchsummary import summary

    mod = VT_AE()#.cuda()
    print(mod)
    # summary(mod, (3,512,512))


