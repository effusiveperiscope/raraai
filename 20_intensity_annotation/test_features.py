from dataset import dataset2, WhisperContext, PedalboardContext
import time
from commons import sequence_mask
import torch


if __name__ == '__main__':
    data = dataset2(filelist='data/v0.1/train_filelist.txt', is_train=True)
    loader = data.loader(batch_size=4, num_workers=0)
    for i in range(5):
        next(iter(loader)) # warmup

    start_time = time.time()
    batch = next(iter(loader))
    end_time = time.time()
    print(f"Time to get one batch: {end_time - start_time:.2f} seconds")
    print(batch['wave'].shape)
    print(batch['wave_length'])

    pedalboard_context = PedalboardContext()
    whisper_context = WhisperContext()
    wave = batch['wave']
    wave_lengths = batch['wave_length']
    wave_np = wave.detach().cpu().numpy()
    waves_processed = []
    for i, wave in enumerate(wave_np):
        waves_processed.append(pedalboard_context.process_wave(wave[:wave_lengths[i]]))
    whisper_features, feature_len = whisper_context.extract_features_batched(
        waves_processed)
    print(feature_len)
    print(whisper_features.shape)

    # 32000 -> 200 frames -> 400 frames upsampled

    interp_whisper_features = whisper_context.interp2(whisper_features)
    feature_mask = sequence_mask(feature_len * 2).to(torch.long)
    interp_whisper_features = interp_whisper_features[:, :feature_len.max() * 2, :]
    print(interp_whisper_features.shape)
    print(feature_mask.shape)