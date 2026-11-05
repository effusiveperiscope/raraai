import time
import torch
import psutil
import os
from transformers import AutoTokenizer, AutoModel

def benchmark_embeddings():
    # Larger sample set for realistic benchmarking
    sentences = [
        "Sample data 1: This is a sentence used to test embedding performance.",
        "Sample data 2: Evaluating vectorization model inference speed and VRAM usage.",
        "Sample data 3: Hugging Face transformers lightweight embedding model.",
        "Sample data 4: Text embedding and semantic search tasks in deep learning."
    ] * 25  # Scale up to 100 sentences for a better benchmark

    print(f"--- Environment & Setup ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Track initial CPU memory
    process = psutil.Process(os.getpid())
    cpu_mem_start = process.memory_info().rss / (1024 * 1024) # MB

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')
    model.to(device)
    model.eval()
    load_time = time.perf_counter() - load_start
    print(f"Model loading time: {load_time:.4f} seconds")

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # --- 1. Warmup Run (Crucial for accurate timing by avoiding JIT/kernel setup overhead) ---
    print("Running warmup...")
    with torch.no_grad():
        _ = model(**encoded_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # --- 2. Benchmark Execution & Time ---
    print("Running benchmark...")
    iterations = 10
    
    start_time = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_per_batch = total_time / iterations
    avg_time_per_sentence = avg_time_per_batch / len(sentences)

    # --- 3. Resource Usage Stats ---
    cpu_mem_end = process.memory_info().rss / (1024 * 1024) # MB
    cpu_mem_diff = cpu_mem_end - cpu_mem_start

    print("\n" + "="*30 + " BENCHMARK RESULTS " + "="*30)
    print(f"Total Sentences Processed per Batch: {len(sentences)}")
    print(f"Total Iterations: {iterations}")
    print(f"--- Time Metrics ---")
    print(f"Total Execution Time: {total_time:.4f} seconds")
    print(f"Average Time per Batch: {avg_time_per_batch * 1000:.2f} ms")
    print(f"Average Time per Sentence: {avg_time_per_sentence * 1000:.2f} ms")
    
    print(f"\n--- Memory Metrics ---")
    print(f"CPU RSS Memory Delta: {cpu_mem_diff:.2f} MB (Current: {cpu_mem_end:.2f} MB)")
    
    if device.type == 'cuda':
        peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak GPU Memory Allocated: {peak_gpu_mem:.2f} MB")
    print("="*79)

if __name__ == "__main__":
    benchmark_embeddings()