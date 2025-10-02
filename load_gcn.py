import torch

def fix_shape_mismatches(state_dict, model):
    new_state_dict = {}
    for name, param in state_dict.items():
        if name in model.state_dict():
            if param.shape != model.state_dict()[name].shape:
                # Try transpose if shapes are swapped
                if param.shape[::-1] == model.state_dict()[name].shape:
                    print(f"Transposing weight for {name} from {param.shape} to {model.state_dict()[name].shape}")
                    param = param.t()
                else:
                    print(f"Skipping {name}: checkpoint {param.shape}, model {model.state_dict()[name].shape}")
                    continue
        new_state_dict[name] = param
    return new_state_dict

def load_gcn_checkpoint(model, checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Rename old GCNConv keys
    renamed = {}
    for k, v in state_dict.items():
        if k.startswith("conv") and k.endswith(".weight"):
            new_k = k.replace(".weight", ".lin.weight")
        else:
            new_k = k
        renamed[new_k] = v

    # Fix mismatched shapes (transpose if needed)
    fixed_state_dict = fix_shape_mismatches(renamed, model)

    # Load into model
    model.load_state_dict(fixed_state_dict, strict=False)
    model.to(device)

    print(f"Loaded checkpoint from {checkpoint_path}")

    return model, checkpoint