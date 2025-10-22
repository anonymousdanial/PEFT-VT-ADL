import torch
import matplotlib.pyplot as plt

# Simulate patch embeddings for 5 patches, each of dim 4
patch_embeddings = torch.zeros(1, 5, 4)

# Create a learnable position embedding (like in ViT)
pos_embedding = torch.nn.Parameter(torch.randn(1, 5, 4))

print('Patch embeddings (before):')
print(patch_embeddings)
print('\nPosition embedding:')
print(pos_embedding)

# Add position embedding to patch embeddings
output = patch_embeddings + pos_embedding
print('\nPatch embeddings (after adding position embedding):')
print(output)

# Visualize the position embedding
plt.imshow(pos_embedding.detach().squeeze().numpy(), cmap='viridis')
plt.title('Position Embedding (visualized)')
plt.xlabel('Embedding Dimension')
plt.ylabel('Patch Index')
plt.colorbar()
plt.show()
