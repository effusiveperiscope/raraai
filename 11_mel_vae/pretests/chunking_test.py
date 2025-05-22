import torch
import librosa
import numpy as np
import soundfile as sf
import math
import time
from tqdm.auto import tqdm # Optional: for progress bar
import bigvgan
from meldataset import get_mel_spectrogram
from pathlib import Path
import soundfile as sf

# --- Helper function for Crossfading ---
def crossfade(audio1, audio2, overlap_len_samples):
    """Applies a constant power crossfade to the overlap region."""
    if overlap_len_samples == 0:
        return np.concatenate((audio1, audio2))
    if len(audio1) < overlap_len_samples or len(audio2) < overlap_len_samples:
         raise ValueError("Audio segments shorter than overlap length")

    fade_curve = np.linspace(0, np.pi/2, overlap_len_samples)
    fade_in = np.sin(fade_curve)**2
    fade_out = np.cos(fade_curve)**2

    # Overlap section
    overlap_part1 = audio1[-overlap_len_samples:] * fade_out
    overlap_part2 = audio2[:overlap_len_samples] * fade_in
    crossfaded_overlap = overlap_part1 + overlap_part2

    # Combine non-overlapping parts and the crossfaded section
    output = np.concatenate((audio1[:-overlap_len_samples], crossfaded_overlap, audio2[overlap_len_samples:]))
    return output

# --- Main Processing Function ---
def process_audio_chunked(
    wav_path,
    model,
    device,
    chunk_len_frames, # Desired length of non-overlapping part of MEL chunk
    overlap_len_frames, # Overlap length on EACH side of the MEL chunk
    method='crossfade', # 'crossfade' or 'discard'
    output_path_template="output_{method}_overlap{overlap}.wav",
    batch_size=1 # Process chunks in batches (optional, for potential speedup on GPU)
    ):
    """
    Processes audio using chunked BigVGAN inference.

    Args:
        wav_path (str): Path to the input audio file.
        model: Loaded BigVGAN model.
        device: Torch device.
        chunk_len_frames (int): The number of frames in the central, non-overlapping
                                part of each mel chunk.
        overlap_len_frames (int): The number of frames to overlap on EACH side.
                                  Total chunk length = chunk_len + 2 * overlap.
        method (str): Recombination method: 'crossfade' or 'discard'.
        output_path_template (str): Path template for saving the output wav.
        batch_size (int): Number of chunks to process in parallel per batch.
    """
    print(f"\n--- Processing {wav_path} ---")
    print(f"Chunk method: {method}, Core chunk frames: {chunk_len_frames}, Overlap frames (each side): {overlap_len_frames}")

    sr = model.h.sampling_rate
    hop_size = model.h.hop_size

    if overlap_len_frames < 0:
        raise ValueError("overlap_len_frames cannot be negative")
    if chunk_len_frames <= 0:
        raise ValueError("chunk_len_frames must be positive")

    # 1. Load audio and compute FULL mel spectrogram
    print("Loading audio and computing full mel spectrogram...")
    wav_np, sr_read = librosa.load(wav_path, sr=sr, mono=True)
    if sr_read != sr:
        print(f"Warning: Resampling audio from {sr_read} Hz to {sr} Hz")
    wav_full = torch.FloatTensor(wav_np).unsqueeze(0).to(device) # [1, T_time]

    mel_full = get_mel_spectrogram(wav_full, model.h) # [1, C_mel, T_frame_full]
    n_mels, total_mel_frames = mel_full.shape[1], mel_full.shape[2]
    print(f"Full mel spectrogram shape: {mel_full.shape}")
    print(f"Total mel frames: {total_mel_frames}")

    # --- Optional: Generate baseline without chunking ---
    print("Generating baseline audio (no chunking)...")
    start_time_full = time.time()
    with torch.inference_mode():
        wav_gen_full = model(mel_full) # [1, 1, T_time_gen]
    wav_gen_full_np = wav_gen_full.squeeze(0).squeeze(0).cpu().numpy()
    end_time_full = time.time()
    print(f"Baseline generation took {end_time_full - start_time_full:.2f}s")
    output_path_full = output_path_template.format(method="full", overlap=0)
    sf.write(output_path_full, wav_gen_full_np, sr)
    print(f"Saved baseline audio to {output_path_full}")
    # --- End Baseline ---


    # 2. Prepare chunks
    total_chunk_len_frames = chunk_len_frames + 2 * overlap_len_frames
    step_size_frames = chunk_len_frames # How much we advance for the next chunk

    mel_chunks = []
    start_frame = 0
    while start_frame < total_mel_frames:
        end_frame = start_frame + total_chunk_len_frames
        # Pad the last chunk if it goes beyond the total frames
        # We need padding at the END of the mel spec here.
        # BigVGAN might handle some padding implicitly, but explicit padding is safer.
        current_chunk = mel_full[:, :, start_frame:end_frame]

        pad_size = end_frame - total_mel_frames
        if pad_size > 0:
            actual_frames_in_chunk = current_chunk.shape[2]
            padding_tuple = (0, pad_size) # Pad only on the right

            # Ensure the chunk dimension isn't zero before trying to pad
            if actual_frames_in_chunk == 0:
                 print(f"  Warning: Last chunk slice has 0 frames. Creating zero chunk of target size.")
                 # Create a zero chunk of the target size if the slice was empty
                 target_shape = list(current_chunk.shape)
                 # Calculate the desired frame length for a standard chunk
                 target_frame_len = chunk_len_frames + 2 * overlap_len_frames
                 target_shape[2] = target_frame_len
                 current_chunk = torch.zeros(target_shape, device=current_chunk.device, dtype=current_chunk.dtype)
                 print(f"  Created zero chunk of size {current_chunk.shape}")
            else:
                # Decide padding mode: Use 'reflect' if possible, otherwise 'constant'
                if pad_size >= actual_frames_in_chunk:
                    pad_mode = 'constant'
                    print(f"  Padding last chunk by {pad_size} frames. Input chunk shorter ({actual_frames_in_chunk} frames), using '{pad_mode}' padding.")
                else:
                    pad_mode = 'reflect'
                    print(f"  Padding last chunk by {pad_size} frames using '{pad_mode}'.")

                # Apply padding
                current_chunk = torch.nn.functional.pad(current_chunk, padding_tuple, mode=pad_mode, value=0) # value=0 for constant padding

            # Optional: Verify final chunk shape (should match total_chunk_len_frames ideally)
            # print(f"  Shape after padding: {current_chunk.shape}")
        mel_chunks.append(current_chunk)
        start_frame += step_size_frames

        # Break if the last chunk fully covered the input (no need for more padding)
        if end_frame >= total_mel_frames and pad_size <= 0 :
            break


    num_chunks = len(mel_chunks)
    print(f"Created {num_chunks} mel chunks for processing.")
    if num_chunks == 0:
        print("Warning: No chunks created. Audio might be too short.")
        return

    # 3. Process chunks through the model (batch processing optional)
    print(f"Vocoding chunks (batch size: {batch_size})...")
    start_time_chunked = time.time()
    generated_audio_chunks = []
    with torch.inference_mode():
        for i in tqdm(range(0, num_chunks, batch_size), desc="Vocoding Batches"):
            batch_mel = torch.cat(mel_chunks[i:min(i+batch_size, num_chunks)], dim=0) # [B, C_mel, T_chunk_frame]
            batch_wav_gen = model(batch_mel) # [B, 1, T_chunk_time]
            # Ensure output is detached, moved to CPU, and converted to numpy
            generated_audio_chunks.extend([w.squeeze(0).squeeze(0).cpu().numpy() for w in batch_wav_gen])
    end_time_chunked = time.time()
    print(f"Chunked generation took {end_time_chunked - start_time_chunked:.2f}s")


    # 4. Recombine audio chunks
    print(f"Recombining audio using '{method}' method...")
    overlap_len_samples = overlap_len_frames * hop_size
    chunk_len_samples = chunk_len_frames * hop_size
    final_audio = None

    if method == 'discard':
        if overlap_len_frames == 0:
            print("Warning: Overlap is 0, 'discard' method becomes simple concatenation.")
            final_audio = np.concatenate(generated_audio_chunks)
        else:
            combined_segments = []
            for i, audio_chunk in enumerate(generated_audio_chunks):
                # Expected length based on non-padded mel frames *in this chunk*
                # We need to know the original number of mel frames for this chunk before padding
                original_mel_start = i * step_size_frames
                original_mel_end = min(original_mel_start + total_chunk_len_frames, total_mel_frames)
                original_mel_frames_in_chunk = original_mel_end - original_mel_start
                expected_samples = original_mel_frames_in_chunk * hop_size

                # Trim the generated audio in case the model produced extra samples due to internal padding
                audio_chunk = audio_chunk[:expected_samples]

                start_keep_idx = 0
                end_keep_idx = len(audio_chunk)

                # Discard start overlap (except for the very first chunk)
                if i > 0:
                    start_keep_idx = overlap_len_samples

                # Discard end overlap (except for the very last chunk)
                # The logic needs refinement: we keep the core 'chunk_len_samples' part
                # plus the *relevant* overlap part that connects to the next/previous chunk.

                # Let's try a simpler logic: keep the core chunk + right overlap for all but last
                # keep the core chunk + left overlap for all but first
                # Concatenate the *core* segments.

                core_start_idx = overlap_len_samples if i > 0 else 0 # Start index of the core audio part
                core_end_idx = core_start_idx + chunk_len_samples # End index of the core audio part

                # Ensure indices are within bounds of the actual audio chunk length
                core_start_idx = min(core_start_idx, len(audio_chunk))
                core_end_idx = min(core_end_idx, len(audio_chunk))

                segment_to_keep = audio_chunk[core_start_idx:core_end_idx]

                # Special handling for the first chunk: include the initial overlap
                if i == 0:
                     segment_to_keep = audio_chunk[0:core_end_idx]

                # Special handling for the last chunk: include the final part after the core
                if i == num_chunks - 1:
                    # Correct start index for the last chunk's keep section
                    last_chunk_start_keep = overlap_len_samples if num_chunks > 1 else 0
                    segment_to_keep = audio_chunk[last_chunk_start_keep:]


                # Revised 'discard' logic - keep central part, discard overlap edges
                start_discard = overlap_len_samples if i > 0 else 0
                # End discard depends on whether it's the last chunk
                end_discard = overlap_len_samples if i < num_chunks - 1 else 0

                keep_start = start_discard
                keep_end = len(audio_chunk) - end_discard

                if keep_start >= keep_end : # Handle very short chunks or large overlaps
                    print(f"Warning: Chunk {i} has no samples left after discard ({keep_start} >= {keep_end}). Skipping.")
                    continue

                valid_segment = audio_chunk[keep_start:keep_end]


                # --- Even Simpler Discard: Keep the middle `chunk_len_frames` worth of audio ---
                # This is the most common interpretation, but might miss context transitions
                center_start_sample = overlap_len_samples
                center_end_sample = center_start_sample + chunk_len_samples
                center_end_sample = min(center_end_sample, len(audio_chunk)) # Boundary check

                if i == 0: # First chunk: keep from beginning up to end of its core part + overlap
                    segment = audio_chunk[:center_end_sample]
                elif i == num_chunks - 1: # Last chunk: keep from start of its core part to the end
                     segment = audio_chunk[center_start_sample:]
                else: # Middle chunks: keep only the core center part
                    segment = audio_chunk[center_start_sample:center_end_sample]

                # Ensure segment is not empty
                if len(segment) > 0:
                    combined_segments.append(segment)
                else:
                     print(f"Warning: Segment {i} is empty after discard operation.")


            if combined_segments:
                final_audio = np.concatenate(combined_segments)
            else:
                print("Error: No valid audio segments after discard.")
                final_audio = np.array([])


    elif method == 'crossfade':
        print(f"Recombining audio using '{method}' method (Overlap-Add)...")
        if not generated_audio_chunks:
             print("Error: No audio chunks generated.")
             final_audio = np.array([])
        elif overlap_len_frames == 0:
            print("Overlap is 0, using simple concatenation.")
            final_audio = np.concatenate(generated_audio_chunks)
            # Trim potential extra samples from concatenation if needed
            # (especially if model generates variable length for same mel input)
            expected_len = total_mel_frames * hop_size
            if len(final_audio) > expected_len:
                 print(f"Trimming concatenated audio from {len(final_audio)} to {expected_len}")
                 final_audio = final_audio[:expected_len]
        else:
            # --- Overlap-Add Implementation ---
            print("Applying overlap-add recombination...")
            overlap_len_samples = overlap_len_frames * hop_size
            step_samples = chunk_len_frames * hop_size # How much we advance for each new chunk's core part

            # Calculate expected total length VERY carefully
            # It's based on the total number of *original* mel frames.
            expected_total_samples = total_mel_frames * hop_size

            # Estimate buffer size: Sum of step sizes + length of the last chunk's non-step part
            # A simpler, safer estimate: expected total samples + one chunk length for safety buffer
            chunk_sample_len_approx = len(generated_audio_chunks[0]) # Approx length of one chunk
            buffer_size = expected_total_samples + chunk_sample_len_approx
            final_audio_buffer = np.zeros(buffer_size)

            # Create fade curves (constant power)
            fade_curve = np.linspace(0, np.pi / 2, overlap_len_samples)
            fade_in_curve = np.sin(fade_curve)**2
            fade_out_curve = np.cos(fade_curve)**2

            current_pos_samples = 0
            for i, audio_chunk in enumerate(tqdm(generated_audio_chunks, desc="Overlap-Add")):
                chunk_len = len(audio_chunk)
                start_idx = current_pos_samples
                end_idx = start_idx + chunk_len

                 # Ensure buffer is large enough (should be rare with initial estimate)
                if end_idx > len(final_audio_buffer):
                    print(f"Warning: Resizing overlap-add buffer (chunk {i})")
                    final_audio_buffer.resize(end_idx + chunk_sample_len_approx) # Add more padding

                # --- Apply Fade In/Out directly to the chunk BEFORE adding ---
                windowed_chunk = audio_chunk.copy() # Work on a copy

                if chunk_len < overlap_len_samples:
                     print(f"Warning: Chunk {i} is shorter ({chunk_len}) than overlap ({overlap_len_samples}). Cannot apply full fade.")
                     # Handle short chunks: maybe just add without full fade?
                     # Or apply partial fade if possible? For now, just add it directly.

                else:
                    # Apply fade-in to the start overlap, except for the very first chunk
                    if i > 0:
                        windowed_chunk[:overlap_len_samples] *= fade_in_curve

                    # Apply fade-out to the end overlap, except for the very last chunk
                    if i < num_chunks - 1:
                         # Make sure chunk is long enough for end overlap fade
                         if chunk_len >= overlap_len_samples:
                              windowed_chunk[-overlap_len_samples:] *= fade_out_curve
                         else:
                              # This case is tricky, the fade out might need adjustment
                              # For simplicity here, we might skip fade-out if too short.
                              print(f"Warning: Chunk {i} too short ({chunk_len}) for full fade-out ({overlap_len_samples}). Skipping end fade.")


                # Add the windowed chunk to the buffer at the correct position
                try:
                    final_audio_buffer[start_idx:end_idx] += windowed_chunk
                except ValueError as e:
                     print(f"Error during buffer addition for chunk {i} at indices {start_idx}:{end_idx} (buffer size {len(final_audio_buffer)})")
                     print(f"Chunk shape: {windowed_chunk.shape}")
                     raise e


                # --- Update current_pos_samples for the *next* chunk ---
                # Advance by the step size (core non-overlapping part)
                # But handle the last chunk differently - it doesn't step further
                if i < num_chunks - 1:
                     current_pos_samples += step_samples


            # Trim the final buffer to the expected length based on original mel frames
            print(f"Overlap-add buffer raw length: {len(final_audio_buffer)}")
            if len(final_audio_buffer) > expected_total_samples:
                print(f"Trimming overlap-add result to expected length: {expected_total_samples}")
                final_audio = final_audio_buffer[:expected_total_samples]
            elif len(final_audio_buffer) < expected_total_samples:
                 print(f"Warning: Overlap-add result ({len(final_audio_buffer)}) shorter than expected ({expected_total_samples}). Using result as is.")
                 final_audio = final_audio_buffer
                 # Optionally pad with zeros here if strict length matching is required:
                 # padding_needed = expected_total_samples - len(final_audio_buffer)
                 # final_audio = np.pad(final_audio_buffer, (0, padding_needed))
            else:
                 final_audio = final_audio_buffer


    else:
        raise ValueError("Unknown method: choose 'crossfade' or 'discard'")

    # 5. Trim final audio to match original (or expected vocoder output) length
    # Original length in samples
    original_samples = wav_full.shape[1]
    # Expected length based on *total* mel frames
    expected_samples_from_mel = total_mel_frames * hop_size

    if final_audio is not None and len(final_audio) > 0:
        # Trim to the length expected from the full mel spectrogram
        if len(final_audio) > expected_samples_from_mel:
            print(f"Trimming final audio from {len(final_audio)} to {expected_samples_from_mel} samples (expected length from mel).")
            final_audio = final_audio[:expected_samples_from_mel]
        elif len(final_audio) < expected_samples_from_mel:
             print(f"Warning: Final audio ({len(final_audio)}) is shorter than expected ({expected_samples_from_mel}). Padding might be needed or chunking parameters adjusted.")
             # Optionally pad with zeros if needed:
             # padding_needed = expected_samples_from_mel - len(final_audio)
             # final_audio = np.pad(final_audio, (0, padding_needed))


        # 6. Save the final audio
        output_path = output_path_template.format(method=method, overlap=overlap_len_frames)
        sf.write(output_path, final_audio, sr)
        print(f"Saved chunked audio ({method}, overlap={overlap_len_frames}) to {output_path}")
    else:
         print(f"Failed to generate audio for method '{method}' with overlap {overlap_len_frames}.")


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    input_wav = "test3.flac" # CHANGE THIS
    output_dir = "chunking_test2" # CHANGE THIS (directory will be created)

    # Ensure output directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "output_{method}_overlap{overlap}.wav")

    # --- Chunking Parameters to Test ---
    # Core chunk length (in mel frames). Larger values mean fewer chunks but more memory per chunk.
    # Rule of thumb: ~1-5 seconds of audio worth of frames. E.g. 22050Hz / 256 hop = 86 frames/sec.
    # So, 86*2 = 172 frames for 2 seconds.
    test_chunk_len_frames = [200] # Example: ~2.3 seconds core chunk

    # Overlap length (in mel frames) on EACH side. Needs to be sufficient for model's receptive field.
    # Often requires experimentation. Start with values covering ~100-500ms.
    # E.g., 0.2s * 86 frames/sec = ~17 frames. Try values around this and larger.
    test_overlap_len_frames = [0, 16, 32, 64] # Test no overlap, and increasing overlaps

    # Recombination methods to test
    test_methods = ['discard', 'crossfade']

    # Batch size for vocoder inference (adjust based on GPU memory)
    inference_batch_size = 4

    # --- Load Model (replace with your actual loading) ---
    print("Loading BigVGAN model...")
    model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_256x', use_cuda_kernel=False)
    model.remove_weight_norm()
    model = model.eval().to('cuda')

    # --- Run Tests ---
    if not os.path.exists(input_wav):
         print(f"ERROR: Input audio file not found: {input_wav}")
         print("Please change the 'input_wav' variable.")
    else:
        for chunk_len in test_chunk_len_frames:
            for overlap in test_overlap_len_frames:
                for method in test_methods:
                    # Skip overlap=0 for discard, it's just concatenation
                    if method == 'discard' and overlap == 0:
                        print(f"\nSkipping method='discard' with overlap=0 (same as crossfade w/ overlap=0)")
                        continue
                    # Skip crossfade overlap=0 if already done by discard (or handle explicitly)
                    if method == 'crossfade' and overlap == 0 :
                         # Only run overlap=0 once
                         if 'discard' not in test_methods: # Only run if discard wasn't tested
                              pass # Proceed to run crossfade with overlap 0
                         else:
                             # Check if discard with 0 overlap was already conceptually covered
                             # If both methods are tested, we only need one overlap=0 case
                             if 0 in test_overlap_len_frames:
                                 print(f"\nSkipping method='crossfade' with overlap=0 (already covered)")
                                 continue


                    process_audio_chunked(
                        wav_path=input_wav,
                        model=model,
                        device='cuda',
                        chunk_len_frames=chunk_len,
                        overlap_len_frames=overlap,
                        method=method,
                        output_path_template=output_template,
                        batch_size=inference_batch_size
                    )

        print("\n--- Testing Complete ---")
        print(f"Outputs saved in: {output_dir}")
        print("Listen to the 'output_full_overlap0.wav' (baseline) and compare with the chunked results.")
        print("Pay attention to discontinuities or artifacts at chunk boundaries in the outputs.")