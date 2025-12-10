from dataset import dataset, LiveDataContext
import librosa
import time

if __name__ == '__main__':
    context = LiveDataContext()
    wave, sr = librosa.load('test.wav', sr=16000)

    result, attn = context.extract_features_batched([wave, wave])
    print(result)