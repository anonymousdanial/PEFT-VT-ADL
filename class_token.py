import torch
from einops import repeat

# Example: batch size 1, 2 patches, each patch is a vector of 4
x = torch.tensor([[[1., 2., 3., 4.], [1., 2., 3., 4.]]])  # shape: (1, 2, 4)
print('Patch embeddings (x):', x.shape)
print(x)

# Example class token: shape (1, 1, 4)
cls_token = torch.nn.Parameter(torch.tensor([[[0.5, 0.5, 0.5, 0.5]]]))

# Repeat class token for batch size
b = x.shape[0]
cls_tokens = repeat(cls_token, '() n d -> b n d', b=b)
print('\nClass token (repeated):', cls_tokens.shape)
print(cls_tokens)

# Concatenate class token to patch embeddings
x_cat = torch.cat((cls_tokens, x), dim=1)
print('\nAfter concatenation:', x_cat.shape)
print(x_cat)

# Now x_cat[0, 0, :] is the class token, x_cat[0, 1, :] and x_cat[0, 2, :] are the patch embeddings
