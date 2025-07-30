import torch

def print_memory_usage():
    # Allocated memory (actively used by tensors)
    allocated = torch.cuda.memory_allocated() / 1024**2  # in MB

    # Reserved memory (memory reserved by the caching allocator)
    reserved = torch.cuda.memory_reserved() / 1024**2  # in MB

    print(f"Allocated: {allocated:.2f} MB")
    print(f"Reserved: {reserved:.2f} MB")