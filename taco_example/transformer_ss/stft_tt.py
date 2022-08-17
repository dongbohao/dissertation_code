import numpy as np
import librosa
import matplotlib.pyplot as plt

y, sr = librosa.load(librosa.ex('trumpet'))
D = librosa.stft(y)
y_hat = librosa.istft(D)

plt.plot(y)
plt.show()
print(len(y))
n = len(y)
n_fft = 1024
y_pad = librosa.util.fix_length(y, size=n + n_fft // 2)

D = librosa.stft(y_pad, n_fft=n_fft,hop_length=256,win_length=1024)
print(D.shape)
y_out = librosa.istft(D,n_fft=n_fft,hop_length=256,win_length=1024)

#np.max(np.abs(y - y_out))
plt.plot(y_out)
plt.show()