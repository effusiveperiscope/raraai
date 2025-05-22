import librosa
import numpy as np
import parselmouth
import pyworld as pw
import torch
from svc_helper.pitch.rmvpe import RMVPEModel


# from utils.pitch_utils import interp_f0

PITCH_EXTRACTORS_ID_TO_NAME = {
    1: 'parselmouth',
    2: 'harvest',
    3: 'rmvpe',
}
PITCH_EXTRACTORS_NAME_TO_ID = {v: k for k, v in PITCH_EXTRACTORS_ID_TO_NAME.items()}


def norm_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = np.log2(f0 + uv)  # avoid arithmetic error
    f0[uv] = -np.inf
    return f0


def denorm_f0(f0, uv, pitch_padding=None):
    f0 = 2 ** f0
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


def interp_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = norm_f0(f0, uv)
    if uv.any() and not uv.all():
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return denorm_f0(f0, uv=None), uv


def get_pitch(pe, wav_data, length, hparams, speed=1, interp_uv=False):
    if pe == 'parselmouth':
        return get_pitch_parselmouth(wav_data, length, hparams, speed=speed, interp_uv=interp_uv)
    elif pe == 'harvest':
        return get_pitch_harvest(wav_data, length, hparams, speed=speed, interp_uv=interp_uv)
    elif pe == 'rmvpe':
        return get_pitch_rmvpe(wav_data, length, hparams, speed=speed, interp_uv=interp_uv)
    else:
        raise ValueError(f" [x] Unknown pitch extractor: {pe}")

from scipy.interpolate import interp1d

def interpolate_f0_length(f0: np.ndarray, target_length: int):
    """
    Interpolates or extrapolates an F0 contour and its corresponding 
    voiced/unvoiced flags to a target length, avoiding boundary artifacts.

    Uses linear interpolation *only* for voiced segments and nearest-neighbor 
    interpolation for voiced/unvoiced flags.

    Args:
        f0 (np.ndarray): The original F0 contour (unvoiced frames should be 0).
        target_length (int): The desired length of the F0 contour.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - The resized F0 contour.
            - The corresponding resized UV flag array (True for unvoiced).
    """
    original_length = f0.shape[0]
    if original_length == target_length:
        uv = f0 == 0
        return f0, uv
        
    # Time axes for original and target lengths
    original_time = np.linspace(0, 1, original_length)
    target_time = np.linspace(0, 1, target_length)

    # --- 1. Interpolate Voiced/Unvoiced status using nearest neighbor ---
    uv_original = (f0 <= 0).astype(np.float32) # Use float for interpolation
    interp_func_uv = interp1d(
        original_time, 
        uv_original, 
        kind='nearest', 
        bounds_error=False, 
        fill_value=(uv_original[0], uv_original[-1]) # Extrapolate using edge UV values
    )
    interpolated_uv_float = interp_func_uv(target_time)
    interpolated_uv = (interpolated_uv_float >= 0.5) # Threshold back to boolean

    # --- 2. Interpolate F0 using linear interpolation *only on voiced frames* ---
    voiced_indices = np.where(f0 > 0)[0]
    
    # Initialize interpolated_f0 with zeros
    interpolated_f0 = np.zeros(target_length, dtype=f0.dtype)

    if len(voiced_indices) >= 2: # Need at least two points for linear interpolation
        f0_voiced = f0[voiced_indices]
        time_voiced = original_time[voiced_indices]
        
        interp_func_f0_voiced = interp1d(
            time_voiced, 
            f0_voiced, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(f0_voiced[0], f0_voiced[-1]) # Extrapolate using first/last *voiced* F0
        )
        
        # Calculate interpolated F0 values *only* for the target frames 
        # that are expected to be voiced based on the nearest-neighbor UV interpolation
        target_voiced_indices = np.where(~interpolated_uv)[0]
        
        if len(target_voiced_indices) > 0:
             # Get the time points corresponding to target voiced frames
             target_time_voiced = target_time[target_voiced_indices]
             # Calculate interpolated F0 values at these specific time points
             interpolated_f0_at_voiced_times = interp_func_f0_voiced(target_time_voiced)
             # Assign these values to the corresponding indices in the final array
             interpolated_f0[target_voiced_indices] = interpolated_f0_at_voiced_times

    elif len(voiced_indices) == 1:
        # Special case: Only one voiced frame. Use nearest neighbor logic for F0 too.
        # Find the target indices closest to the single voiced frame's time.
        single_voiced_time = original_time[voiced_indices[0]]
        single_voiced_f0 = f0[voiced_indices[0]]
        
        # Find target indices where UV interpolation resulted in 'voiced'
        target_voiced_indices = np.where(~interpolated_uv)[0]
        if len(target_voiced_indices) > 0:
             # For simplicity, assign the single F0 value to all target voiced frames.
             # A more complex nearest-neighbor based on time could be done, but might be overkill.
            interpolated_f0[target_voiced_indices] = single_voiced_f0
            
    # Else (no voiced frames): interpolated_f0 remains all zeros, which is correct.

    # --- 3. Ensure consistency ---
    # Double-check: Any frame marked as unvoiced must have F0=0
    interpolated_f0[interpolated_uv] = 0
    # Optional: Any frame with F0 <= 0 should be marked unvoiced
    interpolated_uv = (interpolated_f0 <= 0) 

    return interpolated_f0, interpolated_uv


rmvpeModel = None
rmvpe_hop_size = None
def get_pitch_rmvpe(wav_data, length, hparams, speed=1, interp_uv=False):
    global rmvpeModel, rmvpe_hop_size
    temphop_size = int(np.round(hparams['hop_size'] * speed
        * RMVPEModel.expected_sample_rate / hparams['audio_sample_rate']))

    if rmvpeModel is None or rmvpe_hop_size != temphop_size:
        rmvpe_hop_size = temphop_size
        rmvpeModel = RMVPEModel(device='cuda' if torch.cuda.is_available() else 'cpu',
            hop_length=rmvpe_hop_size)

    resampled_wav_data = librosa.resample(wav_data, 
        orig_sr=hparams['audio_sample_rate'], 
        target_sr=RMVPEModel.expected_sample_rate)

    f0 = rmvpeModel.extract_pitch(torch.from_numpy(resampled_wav_data))

    if f0.size != length:
        f0, uv = interpolate_f0_length(f0, length)
    else:
        uv = f0 == 0
    if uv.any() and interp_uv:
        f0, uv = interp_f0(f0, uv)

    return f0, uv
   
def get_pitch_parselmouth(wav_data, length, hparams, speed=1, interp_uv=False):
    """

    :param wav_data: [T]
    :param length: Expected number of frames
    :param hparams:
    :param speed: Change the speed
    :param interp_uv: Interpolate unvoiced parts
    :return: f0, uv
    """
    hop_size = int(np.round(hparams['hop_size'] * speed))
    time_step = hop_size / hparams['audio_sample_rate']
    f0_min = hparams['f0_min']
    f0_max = hparams['f0_max']

    l_pad = int(np.ceil(1.5 / f0_min * hparams['audio_sample_rate']))
    r_pad = hop_size * ((len(wav_data) - 1) // hop_size + 1) - len(wav_data) + l_pad + 1
    wav_data = np.pad(wav_data, (l_pad, r_pad))

    # noinspection PyArgumentList
    s = parselmouth.Sound(wav_data, sampling_frequency=hparams['audio_sample_rate']).to_pitch_ac(
        time_step=time_step, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max)
    assert np.abs(s.t1 - 1.5 / f0_min) < 0.001
    f0 = s.selected_array['frequency'].astype(np.float32)
    if len(f0) < length:
        f0 = np.pad(f0, (0, length - len(f0)))
    f0 = f0[: length]
    uv = f0 == 0
    if uv.any() and interp_uv:
        f0, uv = interp_f0(f0, uv)
    return f0, uv

def get_pitch_harvest(wav_data, length, hparams, speed=1, interp_uv=False):
    hop_size = int(np.round(hparams['hop_size'] * speed))
    time_step = 1000 * hop_size / hparams['audio_sample_rate']
    f0_floor = hparams['f0_min']
    f0_ceil = hparams['f0_max']

    f0, _ = pw.harvest(wav_data.astype(np.float64), hparams['audio_sample_rate'], f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=time_step)
    f0 = f0.astype(np.float32)

    if f0.size < length:
        f0 = np.pad(f0, (0, length - f0.size))
    f0 = f0[:length]
    uv = f0 == 0
    if uv.any() and interp_uv:
        f0, uv = interp_f0(f0, uv)
    return f0, uv

if __name__ == '__main__':
    import soundfile as sf
    import librosa
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    hparams = {
        'audio_sample_rate': 44100,
        'audio_num_mel_bins': 128,
        'hop_size': 512,
        'f0_min': 65,
        'f0_max': 1100,
    }

    audio_file = 'testlong.flac'
    wav_data, sr = sf.read(audio_file)
    wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=hparams['audio_sample_rate'])

    f0, uv = get_pitch('parselmouth', wav_data, 
        length=np.round(wav_data.shape[0] / hparams['hop_size']).astype(int), hparams=hparams)
    f02, uv2 = get_pitch('harvest', wav_data, 
        length=np.round(wav_data.shape[0] / hparams['hop_size']).astype(int), hparams=hparams)
    f03, uv3 = get_pitch('rmvpe', wav_data, 
        length=np.round(wav_data.shape[0] / hparams['hop_size']).astype(int), hparams=hparams)

    print("F0 extraction complete. Preparing plot...")
    print(f"Shapes - f0: {f0.shape}, f02: {f02.shape}, f03: {f03.shape}")

    # --- Plotting Code ---
    num_frames = f0.shape[0] # Assuming all f0 arrays have the same shape
    rmvpe_hop_size = hparams['hop_size']
    sample_rate = hparams['audio_sample_rate'] # Use SR from hparams for time axis calculation

    # Calculate the time axis for the F0 frames
    time_axis = np.arange(num_frames) * rmvpe_hop_size / sample_rate

    # Create copies and replace 0 Hz (often indicating unvoiced) with NaN
    # This prevents plotting zeros and creates gaps for unvoiced sections.
    f0_plot = f0.copy()
    f0_plot[f0_plot < hparams['f0_min']] = np.nan # Also treat below f0_min as NaN/unvoiced

    f02_plot = f02.copy()
    f02_plot[f02_plot < hparams['f0_min']] = np.nan

    f03_plot = f03.copy()
    f03_plot[f03_plot < hparams['f0_min']] = np.nan

    # Create the plot
    plt.figure(figsize=(18, 6)) # Use a wider figure for potentially long audio

    # Plot each F0 contour
    plt.plot(time_axis, f0_plot, label='parselmouth', linewidth=1.5, alpha=0.8)
    plt.plot(time_axis, f02_plot, label='harvest', linewidth=1.5, alpha=0.7, linestyle='--')
    plt.plot(time_axis, f03_plot, label='rmvpe', linewidth=1.5, alpha=0.6, linestyle=':')

    # Add labels and title
    plt.xlabel("Time (s)")
    plt.ylabel("Fundamental Frequency (Hz)")
    plt.title(f"F0 Contour Comparison ({os.path.basename(audio_file)})")

    # Add legend
    plt.legend()

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.5)

    # Optional: Set reasonable Y-axis limits
    # plt.ylim([0, hparams['f0_max'] * 1.1]) # Start y-axis from 0
    plt.ylim([max(0, hparams['f0_min'] - 50), hparams['f0_max'] + 50]) # Zoom slightly around F0 range


    # Ensure layout is tight
    plt.tight_layout()

    # Display the plot
    print("Displaying plot...")
    plt.show()
    print("Plot closed.")
