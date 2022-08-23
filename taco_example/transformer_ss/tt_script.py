import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt

signal, sr = librosa.load(r'D:\\tacotron2\\DeepLearningExamples\\PyTorch\\SpeechSynthesis\\Tacotron2\\LJSpeech-1.1/wavs/LJ045-0096.wav')

hop_length = 256
n_fft = 512
win_length = 512

# Short-time Fourier Transformation on our audio data
#audio_stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft, win_length = win_length)
# gathering the absolute values for all values in our audio_stft
#spectrogram = np.abs(audio_stft)
#print(spectrogram.shape)
# Converting the amplitude to decibels
#log_spectro = librosa.amplitude_to_db(spectrogram)

log_spectro = librosa.feature.melspectrogram(signal,n_fft=n_fft,hop_length=hop_length,win_length=win_length,n_mels=80)
print(log_spectro)
log_spectro = librosa.amplitude_to_db(log_spectro)

print(log_spectro.shape)
# Plotting the short-time Fourier Transformation
plt.figure(figsize=(20, 5))
# Using librosa.display.specshow() to create our spectrogram
librosa.display.specshow(log_spectro, sr=sr, x_axis='time', y_axis='hz', hop_length=hop_length, cmap='magma',fmax=80)
plt.colorbar(label='Decibels')
plt.title('Spectrogram (dB)', fontdict=dict(size=18))
plt.xlabel('Time', fontdict=dict(size=15))
plt.ylabel('Frequency', fontdict=dict(size=15))
plt.show()


# # Creating a Discrete-Fourier Transform with our FFT algorithm
# fast_fourier_transf = librosa.core.stft(signal, hop_length=256, n_fft=159)
# print(fast_fourier_transf.shape)
# # Magnitudes indicate the contribution of each frequency
# magnitude = np.abs(fast_fourier_transf)
# print(len(magnitude))
# print(sr)
# # mapping the magnitude to the relative frequency bins
# frequency = np.linspace(0, sr, len(magnitude))
# # We only need the first half of the magnitude and frequency
# left_mag = magnitude[:int(len(magnitude)/2)]
# left_freq = frequency[:int(len(frequency)/2)]
# plt.plot(left_freq, left_mag)
# plt.title('Discrete-Fourier Transform', fontdict=dict(size=15))
# plt.xlabel('Frequency', fontdict=dict(size=12))
# plt.ylabel('Magnitude', fontdict=dict(size=12))
# plt.show()