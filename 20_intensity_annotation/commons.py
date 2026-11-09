import torch

def load_state_dict_mismatch(model, state_dict, quiet=False, strict=False):
    """Copy `state_dict` into `model`, skipping tensors that don't match.

    strict=True raises instead of silently skipping (use this in eval scripts:
    a silently skipped `in_proj` means you are evaluating a partly random model).
    """
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

    missing_keys = [k for k in model_state_dict if k not in filtered_state_dict]

    if strict and (mismatched_keys or missing_keys):
        raise RuntimeError(
            "Checkpoint does not match model: "
            f"{len(mismatched_keys)} mismatched/unknown and {len(missing_keys)} missing keys.\n"
            "Mismatched: "
            + ", ".join(f"{k} ckpt={a} model={b}" for k, a, b in mismatched_keys[:8])
            + ("\nMissing: " + ", ".join(missing_keys[:8]) if missing_keys else "")
        )

    if (mismatched_keys or missing_keys) and not quiet:
        print("Mismatched or missing keys (skipped):")
        for key, shape_ckpt, shape_model in mismatched_keys:
            print(f"{key}: checkpoint shape = {shape_ckpt}, model shape = {shape_model}")
        for key in missing_keys:
            print(f"{key}: missing from checkpoint")

    model.load_state_dict(filtered_state_dict, strict=False)
    return mismatched_keys, missing_keys

def load_submodule_prefix(model, prefix : str, state_dict: dict, quiet=False, strict=False):
    state_dict = {
        k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    return load_state_dict_mismatch(model, state_dict, quiet, strict)

def infer_in_channels(state_dict, prefix='model.'):
    """Read the input feature width (whisper d_model) straight out of a checkpoint."""
    key = f'{prefix}in_proj.weight'
    if key not in state_dict:
        raise KeyError(f'{key} not found in checkpoint; cannot infer in_channels')
    return int(state_dict[key].shape[1])

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)