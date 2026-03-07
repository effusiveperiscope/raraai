import torch
import soundfile as sf
from commons import plot_spectrogram

def scale_gradients(module, scale):
    """
    Multiplies the gradients of all parameters in the given module by `scale`
    right after they are computed during backward().
    """
    def hook(module, grad_input, grad_output):
        # grad_input is a tuple of gradients w.r.t. the input tensors
        # grad_output is a tuple of gradients w.r.t. the output tensors
        return tuple(g * scale if g is not None else None for g in grad_input)

    handle = module.register_full_backward_hook(hook)
    return handle

def print_memory_usage():
    # Allocated memory (actively used by tensors)
    allocated = torch.cuda.memory_allocated() / 1024**2  # in MB

    # Reserved memory (memory reserved by the caching allocator)
    reserved = torch.cuda.memory_reserved() / 1024**2  # in MB

    print(f"Allocated: {allocated:.2f} MB")
    print(f"Reserved: {reserved:.2f} MB")

def dump_batched_audio(audio : torch.Tensor, 
    prefix : str = "audio_",
    sr : int = 48000):
    # audio [b, 1, t]
    for i,audio in enumerate(audio):
        audio = audio.squeeze(1).detach().cpu().numpy()
        sf.write(f"{prefix}{i}.wav", audio.squeeze(), sr)

def dump_batched_spectrogram(spec : torch.Tensor,
    prefix : str = "spec_"):
    spec = spec.transpose(1,2)
    for i,spec in enumerate(spec):
        plot_spectrogram(spec.detach().cpu(), save_path=f"{prefix}{i}.png")

def check_param_updates(model, tag):
    print(f"[{tag}] Checking for updated parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f" - {name}: ❌ No grad")
            elif torch.all(param.grad == 0):
                print(f" - {name}: ⚠️ Grad is all zero")
            else:
                print(f" - {name}: ✅ Grad exists")