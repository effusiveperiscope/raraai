import torch
def load_state_dict_mismatch(model, state_dict, quiet=False):
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    mismatched_keys = []

    for key in state_dict:
        if key in model_state_dict:
            if state_dict[key].shape == model_state_dict[key].shape:
                filtered_state_dict[key] = state_dict[key]
            else:
                mismatched_keys.append((key, state_dict[key].shape, model_state_dict[key].shape))
        else:
            mismatched_keys.append((key, state_dict[key].shape, None))  # Key not in model

    if mismatched_keys and not quiet:
        print("Mismatched or missing keys (skipped):")
        for key, shape_ckpt, shape_model in mismatched_keys:
            print(f"{key}: checkpoint shape = {shape_ckpt}, model shape = {shape_model}")

    model.load_state_dict(filtered_state_dict, strict=False)

def load_submodule_prefix(model, prefix : str, state_dict: dict, quiet=False):
    state_dict = {
        k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    load_state_dict_mismatch(model, state_dict, quiet)

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)