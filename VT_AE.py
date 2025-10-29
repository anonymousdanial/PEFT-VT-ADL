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

MODIFICATION: Handles patch size mismatches by spatially pooling when pretrained model uses smaller patches.
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
        # NOTE: This assumes num_patches + 1 (including CLS token)
        self.Digcap = S.DigitCaps(in_num_caps=((image_size//patch_size)**2 + 1)*8*8, in_dim_caps=8)
        self.mask = torch.ones(1, image_size//patch_size, image_size//patch_size).bool().cuda()
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

    For models without a CLS token (like BEiT), a learnable CLS token is added to maintain
    compatibility with the original architecture.
    
    IMPORTANT: If the pretrained model uses a different patch size than requested, this wrapper
    will spatially pool/aggregate the patches to match the target patch size.

    The forward signature matches the original: forward(img, mask=None) -> (batch, num_patches + 1, target_dim)
    """
    def __init__(self, image_size: int, patch_size: int, target_dim: int, depth: int, heads: int, mlp_dim: int, pretrained_name: str = None):
        super().__init__()
        self.image_size = image_size
        self.target_patch_size = patch_size

        if pretrained_name is not None:
            # load pretrained ViT encoder
            self.model = ViTModel.from_pretrained(pretrained_name)
            config = self.model.config
            
            # Get the actual patch size from pretrained model
            self.pretrained_patch_size = config.patch_size if hasattr(config, 'patch_size') else patch_size
            
            print(f"\nLoaded pretrained model: {pretrained_name}")
            print(f"  Pretrained patch size: {self.pretrained_patch_size}")
            print(f"  Target patch size: {self.target_patch_size}")
            
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
            self.pretrained_patch_size = patch_size

        self.emb_dim = self.model.config.hidden_size

        # If the pretrained model was trained with a different image/patch grid size, resize its pos embeddings.
        if (hasattr(self.model.config, 'image_size') and self.model.config.image_size is not None):
            trained_image_size = self.model.config.image_size
            if trained_image_size != image_size:
                try:
                    self._resize_pos_emb(trained_image_size)
                except Exception as e:
                    print(f"Warning: Could not resize positional embeddings: {e}")
                    # If resizing fails, keep the original pos embeddings. Downstream may still work but
                    # could lead to a mismatch in attention behavior.
                    pass

        # Check if model has a CLS token
        # BEiT models typically don't have CLS tokens, so we need to add one
        self.has_native_cls = self._check_has_cls_token()
        
        if not self.has_native_cls:
            # Add a learnable CLS token for models without one (like BEiT)
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_dim))
            print(f"  Added learnable CLS token")
        
        # Handle patch size mismatch by spatial pooling
        self.needs_spatial_pooling = (self.pretrained_patch_size != self.target_patch_size)
        if self.needs_spatial_pooling:
            # Calculate pooling factor
            self.pool_factor = self.target_patch_size // self.pretrained_patch_size
            if self.target_patch_size % self.pretrained_patch_size != 0:
                raise ValueError(f"Target patch size {self.target_patch_size} must be a multiple of pretrained patch size {self.pretrained_patch_size}")
            
            print(f"  Will spatially pool patches with factor {self.pool_factor}×{self.pool_factor}")
            print(f"  Original: {(image_size//self.pretrained_patch_size)**2} patches → Target: {(image_size//self.target_patch_size)**2} patches")

        # Project embedding dimension to target_dim if needed
        if self.emb_dim != target_dim:
            self.proj = nn.Linear(self.emb_dim, target_dim)
        else:
            self.proj = nn.Identity()

    def _check_has_cls_token(self):
        """Check if the model uses a CLS token by examining the embeddings."""
        # Standard ViT models have position embeddings of size (1, num_patches + 1, dim)
        # BEiT models have position embeddings of size (1, num_patches, dim)
        if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'position_embeddings'):
            pos_emb_shape = self.model.embeddings.position_embeddings.shape[1]
            num_patches = (self.image_size // self.pretrained_patch_size) ** 2
            # If position embeddings = num_patches + 1, it has a CLS token
            # If position embeddings = num_patches, it doesn't
            return pos_emb_shape == (num_patches + 1)
        # Default to True if we can't determine
        return True

    def _resize_pos_emb(self, trained_image_size: int):
        """Interpolate the pretrained positional embeddings to match the new image/patch grid."""
        pos_emb = self.model.embeddings.position_embeddings  # (1, seq_len, dim)
        seq_len = pos_emb.shape[1]
        
        # Check if model has CLS token in positional embeddings
        has_cls_in_pos = self._check_has_cls_token()
        
        if has_cls_in_pos:
            num_patches_old = seq_len - 1
            cls_token = pos_emb[:, :1, :]
        else:
            num_patches_old = seq_len
            cls_token = None
            
        old_grid = int(math.sqrt(num_patches_old))
        num_patches_new = (self.image_size // self.pretrained_patch_size) ** 2
        new_grid = int(math.sqrt(num_patches_new))
        
        if old_grid == new_grid:
            return

        if has_cls_in_pos:
            patch_pos = pos_emb[:, 1:, :]
        else:
            patch_pos = pos_emb
            
        # reshape to (1, dim, H, W) for interpolation
        patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(new_grid, new_grid), mode='bilinear', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
        
        if cls_token is not None:
            new_pos = torch.cat([cls_token, patch_pos], dim=1)
        else:
            new_pos = patch_pos
            
        # replace the model's positional embeddings
        self.model.embeddings.position_embeddings = nn.Parameter(new_pos)
        # Ensure the model's config matches the new image/patch grid so
        # the internal embedding computations use the updated sizes.
        # This prevents shape mismatches (for example: pretrained 384-> new 512)
        try:
            self.model.config.image_size = self.image_size
            # patch_size is sometimes an attribute in the HF config
            # Don't update this if we're doing spatial pooling later
            if not self.needs_spatial_pooling:
                self.model.config.patch_size = self.target_patch_size
        except Exception:
            # If config cannot be updated for any reason, we still keep the new pos emb
            pass

    def _spatial_pool_patches(self, x):
        """
        Pool fine-grained patches into coarser patches.
        
        Args:
            x: (batch, num_patches_fine, dim) where num_patches_fine = (H_fine * W_fine)
        
        Returns:
            x_pooled: (batch, num_patches_coarse, dim) where num_patches_coarse = (H_coarse * W_coarse)
        """
        batch_size, num_patches_fine, dim = x.shape
        
        # Calculate grid dimensions
        grid_fine = int(math.sqrt(num_patches_fine))
        grid_coarse = grid_fine // self.pool_factor
        
        # Reshape to spatial grid: (batch, H_fine, W_fine, dim)
        x = x.reshape(batch_size, grid_fine, grid_fine, dim)
        
        # Rearrange to (batch, dim, H_fine, W_fine) for adaptive pooling
        x = x.permute(0, 3, 1, 2)
        
        # Use adaptive average pooling to downsample spatially
        x_pooled = F.adaptive_avg_pool2d(x, (grid_coarse, grid_coarse))
        
        # Rearrange back to (batch, num_patches_coarse, dim)
        x_pooled = x_pooled.permute(0, 2, 3, 1).reshape(batch_size, grid_coarse * grid_coarse, dim)
        
        return x_pooled

    def forward(self, img, mask=None):
        # The HF ViTModel expects `pixel_values` shaped [batch, channels, height, width]
        outputs = self.model(pixel_values=img)
        x = outputs.last_hidden_state  # (b, seq_len, emb_dim)
        
        if self.has_native_cls:
            # Model already has CLS token at position 0, separate it
            cls_token = x[:, :1, :]
            patch_tokens = x[:, 1:, :]
        else:
            # Model doesn't have CLS token (like BEiT), use our learnable one
            batch_size = x.shape[0]
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            patch_tokens = x
        
        # Apply spatial pooling if needed (to match target patch size)
        if self.needs_spatial_pooling:
            patch_tokens = self._spatial_pool_patches(patch_tokens)
        
        # Concatenate CLS token with patch tokens
        x = torch.cat([cls_token, patch_tokens], dim=1)
        
        # Now x always has shape (batch, num_patches_target + 1, emb_dim)
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
        noise = n.sample(latent.size()).squeeze(-1).cuda()
        latent = latent + noise
        return latent

    if noise_type == "speckle":
        noise = torch.randn(latent.size()).cuda()
        latent = latent + latent * noise
        return latent

if __name__ == "__main__":
    from torchsummary import summary

    mod = VT_AE().cuda()
    print(mod)
    # summary(mod, (3,512,512))