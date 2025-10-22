import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# -------------------------------
# DigitCaps class from spatial
# -------------------------------
class Config:
    USE_CUDA = False  # set True if using GPU

class DigitCaps(nn.Module):
    def __init__(self, out_num_caps=2, in_num_caps=4, in_dim_caps=8, out_dim_caps=16, decode_idx=-1):
        super(DigitCaps, self).__init__()
        self.conf = Config()
        self.in_dim_caps = in_dim_caps
        self.in_num_caps = in_num_caps
        self.out_dim_caps = out_dim_caps
        self.out_num_caps = out_num_caps
        self.decode_idx = decode_idx
        self.W = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))

    def forward(self, x):
        # x: [batch, in_num_caps, in_dim_caps]
        x_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        x_hat_detached = x_hat.detach()
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps))
        if self.conf.USE_CUDA:
            b = b.cuda()

        num_iters = 3
        for i in range(num_iters):
            c = F.softmax(b, dim=1)
            if i == num_iters -1:
                outputs = self.squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
            else:
                outputs = self.squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        outputs = torch.squeeze(outputs, dim=-2)

        # Masking for reconstruction
        if self.decode_idx == -1:
            classes = torch.sqrt((outputs ** 2).sum(2))
            classes = F.softmax(classes, dim=1)
            _, max_length_indices = classes.max(dim=1)
        else:
            max_length_indices = torch.ones(outputs.size(0)).long() * self.decode_idx
            if self.conf.USE_CUDA:
                max_length_indices = max_length_indices.cuda()

        masked = Variable(torch.sparse.torch.eye(self.out_num_caps))
        if self.conf.USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices)

        t = (outputs * masked[:, :, None]).sum(dim=1).unsqueeze(1)
        return t, outputs

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

# -------------------------------
# Demo: simulate transformer output
# -------------------------------
if __name__ == "__main__":
    batch_size = 2
    num_tokens = 4    # 1 CLS + 3 patches
    token_dim = 8

    # Random transformer output
    transformer_output = torch.randn(batch_size, num_tokens, token_dim)
    print("Transformer output shape:", transformer_output.shape)

    # Create DigitCaps layer
    digit_caps = DigitCaps(out_num_caps=2, in_num_caps=num_tokens, in_dim_caps=token_dim, out_dim_caps=16)

    # Forward pass
    reconstruction_vector, all_capsules = digit_caps(transformer_output)

    print("\nAll output capsules shape:", all_capsules.shape)
    print("Reconstruction vector shape:", reconstruction_vector.shape)
    print("\nReconstruction vector:", reconstruction_vector)
