"""
peft.py
-------
Parameter-Efficient Fine-Tuning (PEFT) for Vision Transformer (ViT).
Implements a simple LoRA (Low-Rank Adaptation) module for linear layers.
"""
import torch
import torch.nn as nn

class LoraLinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # LoRA parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.randn(out_features, r) * 0.01)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        result = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_out = torch.nn.functional.linear(x, self.lora_A)
            lora_out = torch.nn.functional.linear(lora_out, self.lora_B) * self.alpha / self.r
            result = result + lora_out
        return result

# Utility to replace nn.Linear with LoraLinear in a module

def apply_lora(module, r=4, alpha=1.0):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            lora_layer = LoraLinear(child.in_features, child.out_features, r=r, alpha=alpha, bias=child.bias is not None)
            setattr(module, name, lora_layer)
        else:
            apply_lora(child, r=r, alpha=alpha)
    return module
