import torch
import torch.nn as nn

# Fake image: 20x20x3
image = torch.arange(20*20*3).float().reshape(3, 20, 20)  # (C,H,W)

# Parameters
P = 10   # patch size
C = 3    # channels
D = 8    # embedding dimension

# Step 1: Split into patches (unfold = sliding window)
patches = image.unfold(1, P, P).unfold(2, P, P)  # shape: (C, num_patches_H, num_patches_W, P, P)
patches = patches.permute(1, 2, 0, 3, 4)         # (num_patches_H, num_patches_W, C, P, P)
print("Patches shape:", patches.shape)  # (2,2,3,10,10)

# Step 2: Flatten each patch
patches = patches.reshape(-1, P*P*C)  # (num_patches, patch_dim)
print("Flattened patches shape:", patches.shape)  # (4, 300)

# Step 3: Embedding (Linear projection)
embedding_layer = nn.Linear(P*P*C, D, bias=False)
embedded_patches = embedding_layer(patches)  # (4, D)
print("Embedded patches shape:", embedded_patches.shape)  # (4, 8)

# Show one example
print("\nOne flattened patch (first 10 values):")
print(patches[0][:10])

print("\nEmbedded version of that patch:")
print(embedded_patches[0])
