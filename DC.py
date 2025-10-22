import torch
import torch.nn as nn
from spatial import DigitCaps

# Create example transformer output
# Format: [batch_size, num_tokens, embedding_dim]
# Typically transformer outputs: [CLS token, patch1, patch2, ...]
batch_size = 2
num_tokens = 3  # [CLS], [1,2,3,4], [1,2,3,4]
embedding_dim = 768  # Standard ViT embedding dimension

# Simulate transformer output
transformer_output = torch.randn(batch_size, num_tokens, embedding_dim)
print(f"Transformer output shape: {transformer_output.shape}")
print(f"Example values (first sample, first token): {transformer_output[0, 0, :5]}")

# The DigitCaps class expects input of shape: [batch, in_num_caps, in_dim_caps]
# We need to reshape the transformer output to match this format

# Option 1: Use all tokens (including CLS)
# Reshape to treat each token's embedding as multiple capsules
in_dim_caps = 8  # Dimension of each capsule
in_num_caps = (num_tokens * embedding_dim) // in_dim_caps  # Total number of capsules

# Reshape: [batch, num_tokens, embedding_dim] -> [batch, in_num_caps, in_dim_caps]
x = transformer_output.reshape(batch_size, in_num_caps, in_dim_caps)
print(f"\nReshaped for DigitCaps: {x.shape}")

# Initialize DigitCaps
# out_num_caps: number of output capsules (e.g., number of classes)
# in_num_caps: number of input capsules from transformer
# in_dim_caps: dimension of each input capsule
# out_dim_caps: dimension of each output capsule
digit_caps = DigitCaps(
    out_num_caps=10,  # e.g., 10 classes for digit recognition
    in_num_caps=in_num_caps,
    in_dim_caps=in_dim_caps,
    out_dim_caps=512,
    decode_idx=-1  # -1 means auto-select the longest vector
)

print(f"\nDigitCaps initialized:")
print(f"  Input: {in_num_caps} capsules of dimension {in_dim_caps}")
print(f"  Output: 10 capsules of dimension 512")

# Forward pass
with torch.no_grad():
    selected_output, all_outputs = digit_caps(x)

print(f"\nOutput shapes:")
print(f"  Selected capsule: {selected_output.shape}")  # [batch, 1, out_dim_caps]
print(f"  All capsules: {all_outputs.shape}")  # [batch, out_num_caps, out_dim_caps]

# Compute class probabilities from capsule lengths
capsule_lengths = torch.sqrt((all_outputs ** 2).sum(dim=2))
class_probs = torch.softmax(capsule_lengths, dim=1)

print(f"\nClass probabilities (based on capsule lengths):")
print(f"  Shape: {class_probs.shape}")
print(f"  Sample 1: {class_probs[0]}")
print(f"  Predicted class for sample 1: {class_probs[0].argmax()}")

# ============================================
# Alternative: Use only patch tokens (exclude CLS)
# ============================================
print("\n" + "="*60)
print("Alternative approach: Exclude CLS token")
print("="*60)

# Remove CLS token (first token)
patch_tokens = transformer_output[:, 1:, :]  # [batch, num_patches, embedding_dim]
print(f"Patch tokens shape: {patch_tokens.shape}")

# Reshape patches
num_patches = patch_tokens.shape[1]
in_num_caps_patches = (num_patches * embedding_dim) // in_dim_caps
x_patches = patch_tokens.reshape(batch_size, in_num_caps_patches, in_dim_caps)

# Initialize DigitCaps for patches only
digit_caps_patches = DigitCaps(
    out_num_caps=10,
    in_num_caps=in_num_caps_patches,
    in_dim_caps=in_dim_caps,
    out_dim_caps=512,
    decode_idx=-1
)

print(f"\nDigitCaps for patches only:")
print(f"  Input: {in_num_caps_patches} capsules of dimension {in_dim_caps}")

with torch.no_grad():
    selected_patches, all_patches = digit_caps_patches(x_patches)

print(f"\nPatch-based output shapes:")
print(f"  Selected capsule: {selected_patches.shape}")
print(f"  All capsules: {all_patches.shape}")

# ============================================
# Option 3: Use a projection layer first
# ============================================
print("\n" + "="*60)
print("Option 3: Use projection layer for better control")
print("="*60)

class TransformerToCapsulesAdapter(nn.Module):
    def __init__(self, transformer_dim, in_num_caps, in_dim_caps):
        super().__init__()
        self.projection = nn.Linear(transformer_dim, in_num_caps * in_dim_caps)
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
    
    def forward(self, x):
        # x: [batch, num_tokens, transformer_dim]
        # Use mean pooling over tokens (or just CLS token)
        pooled = x.mean(dim=1)  # [batch, transformer_dim]
        
        # Project to capsule space
        projected = self.projection(pooled)  # [batch, in_num_caps * in_dim_caps]
        
        # Reshape to capsule format
        return projected.reshape(-1, self.in_num_caps, self.in_dim_caps)

# Initialize adapter
desired_in_num_caps = 256
adapter = TransformerToCapsulesAdapter(embedding_dim, desired_in_num_caps, in_dim_caps)

# Initialize DigitCaps with cleaner dimensions
digit_caps_clean = DigitCaps(
    out_num_caps=10,
    in_num_caps=desired_in_num_caps,
    in_dim_caps=in_dim_caps,
    out_dim_caps=512,
    decode_idx=-1
)

print(f"Adapter + DigitCaps pipeline:")
with torch.no_grad():
    capsule_input = adapter(transformer_output)
    print(f"  After adapter: {capsule_input.shape}")
    
    selected_clean, all_clean = digit_caps_clean(capsule_input)
    print(f"  Selected output: {selected_clean.shape}")
    print(f"  All outputs: {all_clean.shape}")

print("\n✓ Example complete!")