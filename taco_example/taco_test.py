import numpy
import torch


print(torch.cuda.is_available())
print(torch.__version__)
#device = torch.device("cpu")
tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
tacotron2 = tacotron2.to("cuda")
tacotron2.eval()


waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to("cuda")
waveglow.eval()

text = "Mrs. De Mohrenschildt thought that Oswald,"

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
sequences, lengths = utils.prepare_input_sequence([text])


mel_spec = []
with torch.no_grad():
    mel, _, _ = tacotron2.infer(sequences, lengths)
    mel_spec.append(mel.to("cpu").squeeze().detach().numpy())
    audio = waveglow.infer(mel)
audio_numpy = audio[0].data.cpu().numpy()
rate = 22050


mel_spec = numpy.vstack(mel_spec)
print("Mel shape",mel_spec.shape)

import librosa
import librosa.display
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 5))
librosa.display.specshow(mel_spec, sr=rate, x_axis='time', y_axis='hz', hop_length=256,
                             cmap='magma', fmax=80)
plt.colorbar(label='Decibels')
plt.savefig("tacotron2.png")


from scipy.io.wavfile import write
write("audio.wav", rate, audio_numpy)

from IPython.display import Audio
Audio(audio_numpy, rate=rate)