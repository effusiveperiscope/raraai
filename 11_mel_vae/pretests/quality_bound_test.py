device = 'cuda'

import torch
import bigvgan
import librosa
from meldataset import get_mel_spectrogram
from pathlib import Path
import soundfile as sf

# instantiate the model. You can optionally set use_cuda_kernel=True for faster inference.
model = bigvgan.BigVGAN.from_pretrained(
    #r'D:\Code\MyBigVGAN'
    'nvidia/bigvgan_v2_44khz_128band_256x'
    , use_cuda_kernel=False)

# remove weight norm in the model and set to eval mode
model.remove_weight_norm()
model = model.eval().to(device)

# load wav file and compute mel spectrogram
def process_audio(wav_path):
    wav, sr = librosa.load(wav_path, sr=model.h.sampling_rate, mono=True)  # wav is np.ndarray with shape [T_time] and values in [-1, 1]
    wav = torch.FloatTensor(wav).unsqueeze(0)  # wav is FloatTensor with shape [B(1), T_time]

    # compute mel spectrogram from the ground truth audio
    mel = get_mel_spectrogram(wav, model.h).to(device)  # mel is FloatTensor with shape [B(1), C_mel, T_frame]
    print(mel.shape)
    print(f"n_fft: {model.h.n_fft}, num_mels: {model.h.num_mels}, sampling_rate: {model.h.sampling_rate}, hop_size: {model.h.hop_size}, win_size: {model.h.win_size}, fmin: {model.h.fmin}, fmax: {model.h.fmax}")

    wav_dur = wav.shape[1] / model.h.sampling_rate
    mel_size_bytes = mel.element_size() * mel.nelement()  # mel_size_bytes is int
    bitrate = 8 * mel_size_bytes / wav_dur  # bitrate is float in bits per second
    print(f"Effective bitrate: {bitrate:.2f} kbps")
    print(f"Frames per second: {mel.shape[2] / wav_dur:.2f}")

    # generate waveform from mel
    with torch.inference_mode():
        wav_gen = model(mel)  # wav_gen is FloatTensor with shape [B(1), 1, T_time] and values in [-1, 1]
    wav_gen_float = wav_gen.squeeze(0).cpu()  # wav_gen is FloatTensor with shape [1, T_time]

    # you can convert the generated waveform to 16 bit linear PCM
    wav_gen_int16 = (wav_gen_float * 32767.0).numpy().astype('int16')  # wav_gen is now np.ndarray with shape [1, T_time] and int16 dtype
    
    in_path = Path(wav_path)
    out_path = in_path.with_name(in_path.stem + '_rec.wav')
    sf.write(out_path, wav_gen_float.numpy().squeeze(), model.h.sampling_rate)

#process_audio('test1.wav')
#process_audio('test2.flac')
process_audio('test3.flac')