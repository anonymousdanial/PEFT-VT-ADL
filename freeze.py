"""
freeze.py
---------
Utility to freeze all parameters in a model except LoRA parameters (lora_A, lora_B).
Usage:
    from freeze import freeze_vit_except_lora
    freeze_vit_except_lora(model)
"""

def freeze_vit_except_lora(model):
    """
    Freezes all parameters in the model except LoRA parameters (lora_A, lora_B).
    Args:
        model (nn.Module): Model containing LoRA layers.
    """
    for name, param in model.named_parameters():
        param.requires_grad = False
    for module in model.modules():
        if hasattr(module, 'lora_A') and module.lora_A is not None:
            module.lora_A.requires_grad = True
        if hasattr(module, 'lora_B') and module.lora_B is not None:
            module.lora_B.requires_grad = True
