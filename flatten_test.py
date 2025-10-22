import torch
from einops import rearrange

# Example: batch of 1 image, 1 channel, 4x4 pixels
tensor = torch.arange(16).reshape(1, 1, 4, 4)
print('Original tensor shape:', tensor.shape)
print('Original tensor:\n', tensor)

# Let's split into 2x2 patches, so patch size = 2
# This will create (4 patches) for a 4x4 image
patch_size = 2
flattened = rearrange(tensor, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
print('After rearrange (patch flatten):', flattened.shape)
print('Flattened patches:\n', flattened)
